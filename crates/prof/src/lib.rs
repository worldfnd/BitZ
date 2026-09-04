//! Per-region wall-clock timing, switched on by an environment variable.
//!
//! [`scope`] returns a guard that records its elapsed time when dropped.
//! Scopes nest: a guard that is alive while an inner guard opens and closes
//! becomes that guard's parent, so each region reports both the time it took
//! including its children and the time it spent outside them.
//!
//! ```no_run
//! fn fold() {}
//! let _guard = prof::scope("prove");
//! {
//!     let _guard = prof::scope("prove/fold");
//!     fold();
//! }
//! ```
//!
//! Set `F2Z_PROFILE` to switch it on, or call [`force_enable`] before the
//! first scope. With it off every [`scope`] call returns an inert guard whose
//! drop does nothing, so instrumented code can stay in the hot path.
//!
//! Timing is per thread. Put scopes on the thread that drives the work, not
//! inside a rayon closure: a scope on the driving thread already covers the
//! parallel section it blocks on, and nesting stays meaningful.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Whether profiling is on, read once.
static ENABLED: OnceLock<bool> = OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var_os("F2Z_PROFILE").is_some())
}

/// Turns the table on regardless of the environment.
///
/// Must run before the first [`scope`]; once the gate has been read this does
/// nothing.
pub fn force_enable() {
    let _ = ENABLED.set(true);
}

/// What one region accumulated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Region {
    pub label: &'static str,
    /// Nesting depth the region was first seen at.
    pub depth: usize,
    /// Index of the enclosing region, which is what makes two scopes sharing
    /// a label under different parents stay separate.
    pub parent: Option<usize>,
    /// Wall-clock time, including nested regions.
    pub inclusive: Duration,
    /// Wall-clock time outside nested regions.
    pub exclusive: Duration,
    pub calls: u64,
}

#[derive(Default)]
struct Table {
    /// In first-seen order, which is execution order for a stable program.
    regions: Vec<Region>,
    /// Open scopes: index into `regions`, when it started, and how much of
    /// that has been spent in children.
    open: Vec<(usize, Instant, Duration)>,
}

thread_local! {
    static TABLE: RefCell<Table> = RefCell::new(Table::default());
}

/// Opens a region. The returned guard closes it when dropped.
pub fn scope(label: &'static str) -> Guard {
    if !enabled() {
        return Guard { active: false };
    }
    TABLE.with(|table| {
        let mut table = table.borrow_mut();
        let depth = table.open.len();
        let parent = table.open.last().map(|open| open.0);
        let index = match table
            .regions
            .iter()
            .position(|region| region.label == label && region.parent == parent)
        {
            Some(index) => index,
            None => {
                table.regions.push(Region {
                    label,
                    depth,
                    parent,
                    inclusive: Duration::ZERO,
                    exclusive: Duration::ZERO,
                    calls: 0,
                });
                table.regions.len() - 1
            }
        };
        table.open.push((index, Instant::now(), Duration::ZERO));
    });
    Guard { active: true }
}

/// Closes the region it was opened for.
#[must_use = "the region closes when the guard drops, so it has to be bound"]
pub struct Guard {
    active: bool,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        TABLE.with(|table| {
            let mut table = table.borrow_mut();
            let Some((index, start, in_children)) = table.open.pop() else {
                return;
            };
            let elapsed = start.elapsed();
            let region = &mut table.regions[index];
            region.inclusive += elapsed;
            region.exclusive += elapsed.saturating_sub(in_children);
            region.calls += 1;
            if let Some(parent) = table.open.last_mut() {
                parent.2 += elapsed;
            }
        });
    }
}

/// Takes every region recorded on this thread and clears the table.
pub fn take() -> Vec<Region> {
    TABLE.with(|table| std::mem::take(&mut table.borrow_mut().regions))
}

/// Prints the regions as a tree and clears the table.
///
/// Shares are against the outermost region's inclusive time, so a top-level
/// region reads 100% and its children divide it.
pub fn dump(title: &str) {
    let regions = take();
    if regions.is_empty() {
        return;
    }
    let total = regions
        .iter()
        .filter(|region| region.depth == 0)
        .map(|region| region.inclusive)
        .sum::<Duration>()
        .as_secs_f64();

    eprintln!("\n{title}");
    eprintln!(
        "{:<44} {:>10} {:>10} {:>7} {:>6}",
        "region", "total s", "self s", "share", "calls"
    );
    for region in &regions {
        let indent = "  ".repeat(region.depth);
        let name = format!("{indent}{}", region.label);
        let share = if total > 0.0 {
            100.0 * region.inclusive.as_secs_f64() / total
        } else {
            0.0
        };
        eprintln!(
            "{name:<44} {:>10.3} {:>10.3} {share:>6.1}% {:>6}",
            region.inclusive.as_secs_f64(),
            region.exclusive.as_secs_f64(),
            region.calls,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One label under two different parents has to stay two regions,
    /// otherwise the prover's and the verifier's identically named scopes
    /// merge and every share is wrong.
    #[test]
    fn the_same_label_under_different_parents_stays_separate() {
        force_enable();
        let _ = take();

        for parent in ["prove", "verify"] {
            let _outer = scope(parent);
            let _inner = scope("shared");
            std::thread::sleep(Duration::from_millis(5));
        }

        let regions = take();
        let shared: Vec<_> = regions.iter().filter(|r| r.label == "shared").collect();
        assert_eq!(shared.len(), 2, "the two parents' scopes were merged");
        assert!(shared.iter().all(|region| region.calls == 1));
    }

    /// Nesting has to split the parent's time into its own and its children's.
    #[test]
    fn a_nested_scope_is_charged_to_its_parent() {
        force_enable();
        let _ = take();

        {
            let _outer = scope("outer");
            {
                let _inner = scope("outer/inner");
                std::thread::sleep(Duration::from_millis(20));
            }
            std::thread::sleep(Duration::from_millis(20));
        }

        let regions = take();
        assert_eq!(regions.len(), 2);
        let (outer, inner) = (regions[0], regions[1]);

        assert_eq!(outer.depth, 0);
        assert_eq!(inner.depth, 1);
        assert!(outer.inclusive >= inner.inclusive);
        assert!(
            outer.exclusive < outer.inclusive,
            "the inner scope's time is not the parent's own",
        );
        assert!(inner.exclusive >= Duration::from_millis(15));
    }

    /// The same label at the same depth accumulates rather than repeating.
    #[test]
    fn repeated_scopes_accumulate() {
        force_enable();
        let _ = take();

        for _ in 0..3 {
            let _guard = scope("repeated");
            std::thread::sleep(Duration::from_millis(5));
        }

        let regions = take();
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].calls, 3);
        assert!(regions[0].inclusive >= Duration::from_millis(12));
    }
}

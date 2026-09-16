// Only the CLI binary installs this layer.

use std::{
    fmt::{self, Write as _},
    io::Write as _,
    time::Instant,
};
use tracing::{
    Event, Subscriber,
    field::{Field, Visit},
    span::{Attributes, Id, Record},
};
use tracing_subscriber::{Layer, fmt::MakeWriter, layer::Context, registry::LookupSpan};

struct Data {
    depth: usize,
    started: Instant,
    updates: String,
}

struct Fields<'a>(&'a mut String);

impl Visit for Fields<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.0, " {value:?}");
        } else {
            let _ = write!(self.0, " {}={value:?}", field.name());
        }
    }
}

pub struct SpanStats<W> {
    writer: W,
    ansi: bool,
}

impl<W> SpanStats<W> {
    pub fn new(writer: W, ansi: bool) -> Self {
        Self { writer, ansi }
    }

    fn dim(&self) -> &str {
        if self.ansi { "\x1b[2m" } else { "" }
    }

    fn reset(&self) -> &str {
        if self.ansi { "\x1b[0m" } else { "" }
    }
}

impl<S, W> Layer<S> for SpanStats<W>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    W: for<'a> MakeWriter<'a> + 'static,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        // A filtered ancestor can exist without data from this layer.
        let depth = span.parent().map_or(0, |parent| {
            parent
                .scope()
                .find_map(|ancestor| {
                    ancestor
                        .extensions()
                        .get::<Data>()
                        .map(|data| data.depth + 1)
                })
                .unwrap_or(0)
        });
        span.extensions_mut().insert(Data {
            depth,
            started: Instant::now(),
            updates: String::new(),
        });
        let mut line = branch(depth, '╮');
        let _ = write!(
            line,
            "{}{}::{}{}",
            self.dim(),
            span.metadata().target(),
            self.reset(),
            span.metadata().name(),
        );
        attrs.record(&mut Fields(&mut line));
        // A closed stderr pipe must not interrupt proof generation.
        let _ = writeln!(self.writer.make_writer(), "{line}");
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id)
            && let Some(data) = span.extensions_mut().get_mut::<Data>()
        {
            values.record(&mut Fields(&mut data.updates));
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let mut line = String::new();
        // Respect explicit parents as well as the thread's current span.
        if let Some(scope) = ctx.event_scope(event) {
            for span in scope {
                if let Some(data) = span.extensions().get::<Data>() {
                    line.push_str(&"│ ".repeat(data.depth + 1));
                    let _ = write!(
                        line,
                        "{}{:.2?}{} ",
                        self.dim(),
                        data.started.elapsed(),
                        self.reset(),
                    );
                    break;
                }
            }
        }
        let color = if self.ansi {
            match *event.metadata().level() {
                tracing::Level::ERROR => "\x1b[1;31m",
                tracing::Level::WARN => "\x1b[1;33m",
                tracing::Level::INFO => "\x1b[1;32m",
                _ => self.dim(),
            }
        } else {
            ""
        };
        let _ = write!(line, "{color}{}{}", event.metadata().level(), self.reset());
        event.record(&mut Fields(&mut line));
        let _ = writeln!(self.writer.make_writer(), "{line}");
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else {
            return;
        };
        let extensions = span.extensions();
        let Some(data) = extensions.get::<Data>() else {
            return;
        };
        let mut line = branch(data.depth, '╯');
        let _ = write!(
            line,
            "{}{}:{} {:.2?}{} duration{}{}",
            self.dim(),
            span.metadata().name(),
            self.reset(),
            data.started.elapsed(),
            self.dim(),
            self.reset(),
            data.updates,
        );
        let _ = writeln!(self.writer.make_writer(), "{line}");
    }
}

fn branch(depth: usize, marker: char) -> String {
    let mut line = String::new();
    if depth > 0 {
        line.push_str(&"│ ".repeat(depth - 1));
        line.push_str("├─");
    }
    let _ = write!(line, "{marker} ");
    line
}

#[cfg(test)]
mod tests {
    use std::{
        io,
        sync::{Arc, Mutex},
    };

    use tracing_subscriber::{filter::filter_fn, prelude::*};

    use super::*;

    #[derive(Clone, Default)]
    struct Buffer(Arc<Mutex<Vec<u8>>>);

    impl io::Write for Buffer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for Buffer {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl Buffer {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    fn capture(action: impl FnOnce()) -> String {
        let buffer = Buffer::default();
        let subscriber = tracing_subscriber::registry().with(SpanStats::new(buffer.clone(), false));
        tracing::subscriber::with_default(subscriber, action);
        buffer.text()
    }

    #[test]
    fn nested_spans_include_fields_events_and_durations_without_ansi() {
        let output = capture(|| {
            let root = tracing::info_span!(target: "test", "prove", size = 8);
            let _root = root.enter();
            let child =
                tracing::info_span!(target: "test", "commit", count = tracing::field::Empty);
            let _child = child.enter();
            child.record("count", 3_u64);
            tracing::info!(proof_bytes = 42, "proof ready");
        });
        let lines: Vec<_> = output.lines().collect();

        assert_eq!(lines.len(), 5, "{output}");
        assert_eq!(lines[0], "╮ test::prove size=8");
        assert_eq!(lines[1], "├─╮ test::commit");
        assert!(lines[2].starts_with("│ │ "), "{output}");
        assert!(
            lines[2].contains("INFO proof ready proof_bytes=42"),
            "{output}"
        );
        assert!(lines[3].starts_with("├─╯ commit: "), "{output}");
        assert!(lines[3].ends_with(" duration count=3"), "{output}");
        assert!(lines[4].starts_with("╯ prove: "), "{output}");
        for line in &lines[3..] {
            let duration = line.split_once(": ").unwrap().1.split(' ').next().unwrap();
            assert!(
                duration.starts_with(|ch: char| ch.is_ascii_digit()),
                "{output}"
            );
            assert!(duration.ends_with('s'), "{output}");
        }
        assert!(!output.contains('\x1b'), "{output}");
    }

    #[test]
    fn events_follow_explicit_parents_and_explicit_roots() {
        let output = capture(|| {
            let root = tracing::info_span!(target: "test", "root");
            let _root = root.enter();
            let child = tracing::info_span!(target: "test", "child");
            let _child = child.enter();
            tracing::info!(parent: &root, "parent event");
            tracing::info!(parent: None, "root event");
            tracing::info!("current event");
        });
        let parent = output
            .lines()
            .find(|line| line.ends_with("parent event"))
            .unwrap();
        let root = output
            .lines()
            .find(|line| line.ends_with("root event"))
            .unwrap();
        let current = output
            .lines()
            .find(|line| line.ends_with("current event"))
            .unwrap();

        assert!(parent.starts_with("│ "), "{output}");
        assert!(!parent.starts_with("│ │ "), "{output}");
        assert_eq!(root, "INFO root event");
        assert!(current.starts_with("│ │ "), "{output}");
    }

    #[test]
    fn filtered_ancestors_do_not_add_tree_depth() {
        // Keep filtered spans in the registry, as another output layer can do.
        struct KeepSpans;
        impl<S: Subscriber> Layer<S> for KeepSpans {}

        let buffer = Buffer::default();
        let subscriber = tracing_subscriber::registry().with(KeepSpans).with(
            SpanStats::new(buffer.clone(), false)
                .with_filter(filter_fn(|metadata| metadata.name() != "hidden")),
        );
        tracing::subscriber::with_default(subscriber, || {
            let root = tracing::info_span!(target: "test", "visible");
            let _root = root.enter();
            let hidden = tracing::info_span!(target: "test", "hidden");
            let _hidden = hidden.enter();
            let child = tracing::info_span!(target: "test", "child");
            let _child = child.enter();
            tracing::info!(parent: &hidden, "filtered parent event");
        });
        tracing::subscriber::with_default(
            tracing_subscriber::registry().with(KeepSpans).with(
                SpanStats::new(buffer.clone(), false)
                    .with_filter(filter_fn(|metadata| metadata.name() != "hidden")),
            ),
            || {
                let hidden = tracing::info_span!(target: "test", "hidden");
                let _hidden = hidden.enter();
                let _child = tracing::info_span!(target: "test", "orphan");
            },
        );
        let output = buffer.text();

        assert!(!output.contains("hidden"), "{output}");
        assert!(
            output.lines().any(|line| line == "├─╮ test::child"),
            "{output}"
        );
        assert!(
            output.lines().any(|line| line == "╮ test::orphan"),
            "{output}"
        );
        let event = output
            .lines()
            .find(|line| line.ends_with("filtered parent event"))
            .unwrap();
        // Context hides this explicit parent. Do not attach the event to the active child.
        assert_eq!(event, "INFO filtered parent event");
    }

    #[test]
    fn writer_errors_do_not_interrupt_spans_or_events() {
        struct BrokenWriter;

        impl io::Write for BrokenWriter {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::ErrorKind::BrokenPipe.into())
            }

            fn flush(&mut self) -> io::Result<()> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
        }

        let subscriber =
            tracing_subscriber::registry().with(SpanStats::new(|| BrokenWriter, false));
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!("proof");
            let _entered = span.enter();
            tracing::warn!("event survives writer failure");
        });
    }
}

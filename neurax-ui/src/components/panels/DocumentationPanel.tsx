/**
 * The user guide, in the application.
 *
 * Documentation that lives on a website is documentation you read once, before
 * you start, and never again — by the time you have a question you are inside
 * the tool with a design on screen, and leaving to find an answer costs more
 * than guessing. So the guide ships in the binary, opens over the workspace
 * with F1, and is searchable, because a reader with a specific question does
 * not want a table of contents.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog.tsx';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Search, BookOpen, Info, AlertTriangle, X } from 'lucide-react';
import {
  DOCUMENTATION,
  DocBlock,
  DocSection,
  searchDocs,
} from '@/data/documentation.ts';

interface DocumentationPanelProps {
  isOpen: boolean;
  onClose: () => void;
  /** Section to open at, when the panel is opened from a specific place. */
  initialSectionId?: string;
}

/**
 * Render the small subset of Markdown the guide uses: `**bold**` and `` `code` ``.
 *
 * A full Markdown renderer would be a dependency and an escaping surface for
 * two kinds of emphasis. This handles exactly what the content contains, and
 * anything it does not recognise renders as the literal text it is.
 */
function RichText({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);

  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return (
            <strong key={i} className="font-semibold text-foreground">
              {part.slice(2, -2)}
            </strong>
          );
        }
        if (part.startsWith('`') && part.endsWith('`')) {
          return (
            <code
              key={i}
              className="rounded bg-secondary px-1 py-0.5 font-mono text-[0.85em] text-foreground"
            >
              {part.slice(1, -1)}
            </code>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

function Block({ block }: { block: DocBlock }) {
  switch (block.kind) {
    case 'heading':
      return (
        <h4 className="mt-6 mb-2 text-sm font-semibold text-foreground">
          <RichText text={block.text} />
        </h4>
      );

    case 'text':
      return (
        <p className="mb-3 text-sm leading-relaxed text-muted-foreground">
          <RichText text={block.text} />
        </p>
      );

    case 'list':
      return (
        <ul className="mb-3 space-y-1.5">
          {block.items.map((item, i) => (
            <li key={i} className="flex gap-2 text-sm leading-relaxed text-muted-foreground">
              <span className="mt-[0.45rem] h-1 w-1 flex-shrink-0 rounded-full bg-muted-foreground/60" />
              <span>
                <RichText text={item} />
              </span>
            </li>
          ))}
        </ul>
      );

    case 'steps':
      return (
        <ol className="mb-3 space-y-2">
          {block.items.map((item, i) => (
            <li key={i} className="flex gap-3 text-sm leading-relaxed text-muted-foreground">
              <span className="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-primary/15 text-[11px] font-semibold text-primary">
                {i + 1}
              </span>
              <span className="pt-0.5">
                <RichText text={item} />
              </span>
            </li>
          ))}
        </ol>
      );

    case 'code':
      return (
        <figure className="mb-4">
          {block.caption && (
            <figcaption className="mb-1 text-xs text-muted-foreground">{block.caption}</figcaption>
          )}
          {/* Wide code scrolls inside its own box; the page never scrolls sideways. */}
          <pre className="overflow-x-auto rounded-md border border-border bg-secondary/40 p-3 text-xs leading-relaxed">
            <code className="font-mono text-foreground">{block.text}</code>
          </pre>
        </figure>
      );

    case 'keys':
      return (
        <div className="mb-4 overflow-hidden rounded-md border border-border">
          {block.items.map((item, i) => (
            <div
              key={item.keys}
              className={`flex items-center gap-4 px-3 py-2 text-sm ${i % 2 ? 'bg-secondary/20' : ''}`}
            >
              <kbd className="min-w-[9rem] flex-shrink-0 font-mono text-xs font-medium text-foreground">
                {item.keys}
              </kbd>
              <span className="text-muted-foreground">
                <RichText text={item.action} />
              </span>
            </div>
          ))}
        </div>
      );

    case 'table':
      return (
        <div className="mb-4 overflow-x-auto rounded-md border border-border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-secondary/40">
                {block.columns.map((column) => (
                  <th
                    key={column}
                    className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground"
                  >
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, i) => (
                <tr key={i} className={i % 2 ? 'bg-secondary/20' : undefined}>
                  {row.map((cell, j) => (
                    <td
                      key={j}
                      className={`px-3 py-2 align-top ${j === 0 ? 'text-foreground' : 'text-muted-foreground'}`}
                    >
                      <RichText text={cell} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );

    case 'note': {
      const warning = block.tone === 'warning';
      const Icon = warning ? AlertTriangle : Info;
      return (
        <aside
          className={`mb-4 rounded-md border p-3 ${
            warning
              ? 'border-amber-500/30 bg-amber-500/10'
              : 'border-primary/25 bg-primary/[0.07]'
          }`}
        >
          <p
            className={`flex items-center gap-1.5 text-xs font-semibold ${
              warning ? 'text-amber-600 dark:text-amber-400' : 'text-primary'
            }`}
          >
            <Icon className="h-3.5 w-3.5 flex-shrink-0" />
            {block.title}
          </p>
          <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
            <RichText text={block.text} />
          </p>
        </aside>
      );
    }
  }
}

function Section({ section }: { section: DocSection }) {
  return (
    <section id={`doc-${section.id}`} className="scroll-mt-4 pb-8">
      <h3 className="mb-1 text-lg font-semibold text-foreground">{section.title}</h3>
      <p className="mb-4 text-sm text-muted-foreground/80">{section.summary}</p>
      {section.blocks.map((block, i) => (
        <Block key={i} block={block} />
      ))}
    </section>
  );
}

export function DocumentationPanel({
  isOpen,
  onClose,
  initialSectionId,
}: DocumentationPanelProps) {
  const [query, setQuery] = useState('');
  const [activeId, setActiveId] = useState<string>(
    initialSectionId ?? DOCUMENTATION[0].sections[0].id,
  );
  const bodyRef = useRef<HTMLDivElement>(null);

  const results = useMemo(() => (query.trim() ? searchDocs(query) : null), [query]);

  // Follow a request to open at a particular section.
  useEffect(() => {
    if (isOpen && initialSectionId) {
      setActiveId(initialSectionId);
      setQuery('');
    }
  }, [isOpen, initialSectionId]);

  // Scrolling the body rather than the section into view keeps the panel's own
  // scroll position predictable — `scrollIntoView` would also scroll the page
  // behind the dialog on some platforms.
  useEffect(() => {
    if (!isOpen) return;
    const target = bodyRef.current?.querySelector(`#doc-${CSS.escape(activeId)}`);
    if (target instanceof HTMLElement && bodyRef.current) {
      bodyRef.current.scrollTop = target.offsetTop - 8;
    }
  }, [activeId, isOpen, results]);

  const activeChapter =
    DOCUMENTATION.find((chapter) => chapter.sections.some((s) => s.id === activeId)) ??
    DOCUMENTATION[0];

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="flex h-[85vh] max-w-5xl flex-col gap-0 overflow-hidden p-0">
        {/* Header */}
        <div className="flex items-center gap-3 border-b border-border px-4 py-3">
          <BookOpen className="h-5 w-5 flex-shrink-0 text-primary" />
          <DialogTitle className="text-base font-semibold">NEURAX guide</DialogTitle>

          <div className="relative ml-auto w-full max-w-xs">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search the guide"
              className="h-8 pl-8 pr-8 text-sm"
              aria-label="Search the guide"
            />
            {query && (
              <button
                type="button"
                onClick={() => setQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                aria-label="Clear search"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>

        <div className="flex min-h-0 flex-1">
          {/* Contents */}
          <nav className="hidden w-64 flex-shrink-0 overflow-y-auto border-r border-border py-3 sm:block">
            {DOCUMENTATION.map((chapter) => (
              <div key={chapter.id} className="mb-4">
                <p className="px-4 pb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                  {chapter.title}
                </p>
                {chapter.sections.map((section) => (
                  <button
                    key={section.id}
                    type="button"
                    onClick={() => {
                      setQuery('');
                      setActiveId(section.id);
                    }}
                    className={`block w-full px-4 py-1.5 text-left text-sm transition-colors ${
                      activeId === section.id && !results
                        ? 'bg-primary/10 font-medium text-primary'
                        : 'text-muted-foreground hover:bg-secondary/40 hover:text-foreground'
                    }`}
                  >
                    {section.title}
                  </button>
                ))}
              </div>
            ))}
          </nav>

          {/* Body */}
          <div
            ref={bodyRef}
            className="relative min-w-0 flex-1 overflow-y-auto px-5 py-4 sm:px-8"
          >
            {results ? (
              <>
                <p className="mb-4 text-sm text-muted-foreground">
                  {results.length === 0
                    ? `Nothing in the guide matches “${query}”.`
                    : `${results.length} ${results.length === 1 ? 'section' : 'sections'} matching “${query}”`}
                </p>
                {results.map((section) => (
                  <div key={section.id}>
                    <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                      {section.chapter}
                    </p>
                    <Section section={section} />
                  </div>
                ))}
              </>
            ) : (
              <>
                <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                  {activeChapter.title}
                </p>
                {activeChapter.sections.map((section) => (
                  <Section key={section.id} section={section} />
                ))}
              </>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center gap-3 border-t border-border px-4 py-2">
          <p className="text-xs text-muted-foreground">
            Press <kbd className="font-mono font-medium text-foreground">F1</kbd> to open this
            guide at any time.
          </p>
          <Button variant="outline" size="sm" className="ml-auto" onClick={onClose}>
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

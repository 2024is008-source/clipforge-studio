import { createFileRoute, ClientOnly } from "@tanstack/react-router";
import Studio from "@/components/Studio";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "ClipForge Studio — AI viral clip maker" },
      { name: "description", content: "Turn long videos into viral vertical shorts with AI moment detection, GPU face tracking and animated captions." },
      { property: "og:title", content: "ClipForge Studio — AI viral clip maker" },
      { property: "og:description", content: "AI moment detection, GPU face tracking and animated captions — rendered on your own graphics card." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Loading() {
  return <div className="grid min-h-screen place-items-center font-mono text-sm text-muted-foreground">Loading studio…</div>;
}

function Index() {
  return (
    <ClientOnly fallback={<Loading />}>
      <Studio />
    </ClientOnly>
  );
}

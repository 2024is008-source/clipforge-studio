import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";

import { cn } from "@/lib/utils";

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, ...props }, ref) => {
  // Radix renders a thumb per value; a single hard-coded thumb silently strips the end handle
  // off every two-value range slider in the app.
  const value = props.value ?? props.defaultValue;
  const thumbs = Array.isArray(value) ? Math.max(1, value.length) : 1;
  // The thumb carries role="slider", so a label on the root never reaches assistive tech.
  const label = props["aria-label"];
  const { "aria-label": _rootLabel, ...rootProps } = props;
  return (
    <SliderPrimitive.Root
      ref={ref}
      className={cn("relative flex w-full touch-none select-none items-center", className)}
      {...rootProps}
    >
      <SliderPrimitive.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-primary/20">
        <SliderPrimitive.Range className="absolute h-full bg-primary" />
      </SliderPrimitive.Track>
      {Array.from({ length: thumbs }, (_, i) => (
        <SliderPrimitive.Thumb
          key={i}
          aria-label={label ? (thumbs > 1 ? `${label} ${i === 0 ? "start" : "end"}` : label) : undefined}
          className="block h-4 w-4 rounded-full border-2 border-background bg-primary transition-transform duration-150 ease-out hover:scale-125 disabled:pointer-events-none disabled:opacity-50"
        />
      ))}
    </SliderPrimitive.Root>
  );
});
Slider.displayName = SliderPrimitive.Root.displayName;

export { Slider };

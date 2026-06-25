#!/usr/bin/env python3
import argparse
import io
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import median


os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


LINE_RE = re.compile(
    r"run=(?P<run>\d+)\s+stride=(?P<stride>\d+)\s+"
    r"separated_by_stride_pages_and_cacheline<STRIDE>_cycles:\s*(?P<cycles>\d+)"
)


def parse_runs(path):
    runs_by_stride = defaultdict(list)
    ignored = 0

    with path.open() as result_file:
        for line in result_file:
            match = LINE_RE.search(line)
            if not match:
                ignored += 1
                continue

            stride = int(match.group("stride"))
            cycles = int(match.group("cycles"))
            runs_by_stride[stride].append(cycles)

    return runs_by_stride, ignored


def median_cycles_by_stride(runs_by_stride, runs_per_stride):
    median_cycles = {}
    incomplete = {}

    for stride, cycles in sorted(runs_by_stride.items()):
        if len(cycles) < runs_per_stride:
            incomplete[stride] = len(cycles)
            continue

        median_cycles[stride] = median(cycles[:runs_per_stride])

    return median_cycles, incomplete


def plot_median_cycles(median_cycles, output_path):
    strides = list(median_cycles)
    cycles = [median_cycles[stride] for stride in strides]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    ax.plot(strides, cycles, marker="o", linewidth=1.8, markersize=4)
    ax.set_xlabel("Stride")
    ax.set_ylabel("Median cycles (billions)")
    ax.set_title("Median Cycles by Stride")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e9:.1f}"))
    fig.tight_layout()
    fig.canvas.draw()

    svg_width, svg_height = (fig.get_size_inches() * 72).tolist()
    axes_bbox = ax.get_position()
    metadata = {
        "axes": {
            "left": axes_bbox.x0 * svg_width,
            "right": axes_bbox.x1 * svg_width,
            "top": (1 - axes_bbox.y1) * svg_height,
            "bottom": (1 - axes_bbox.y0) * svg_height,
        },
        "xlim": list(ax.get_xlim()),
        "ylim": list(ax.get_ylim()),
    }

    interactive = output_path.suffix.lower() == ".svg"
    if interactive:
        write_interactive_svg(fig, output_path, metadata)
    else:
        fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return interactive


def write_interactive_svg(fig, output_path, metadata):
    svg_buffer = io.StringIO()
    fig.savefig(svg_buffer, format="svg")
    svg = svg_buffer.getvalue()
    overlay = coordinate_overlay_svg(metadata)

    if "</svg>" not in svg:
        raise RuntimeError("Matplotlib did not write a complete SVG document")

    output_path.write_text(svg.replace("</svg>", overlay + "\n</svg>", 1), encoding="utf-8")


def coordinate_overlay_svg(metadata):
    axes = metadata["axes"]
    left = axes["left"]
    right = axes["right"]
    top = axes["top"]
    bottom = axes["bottom"]
    width = right - left
    height = bottom - top
    readout_x = left + 8
    readout_y = bottom - 30
    text_y = readout_y + 15
    data_json = json.dumps(metadata)

    return f"""
<g id="coordinate-overlay">
  <defs>
    <clipPath id="coordinate-clip">
      <rect x="{left:.6f}" y="{top:.6f}" width="{width:.6f}" height="{height:.6f}"/>
    </clipPath>
  </defs>
  <style>
    #coordinate-hit-area {{ cursor: crosshair; }}
    #coordinate-crosshair {{ display: none; pointer-events: none; }}
    #coordinate-crosshair line {{ stroke: #dc2626; stroke-width: 0.9; stroke-opacity: 0.8; }}
    #coordinate-readout {{ pointer-events: none; }}
    #coordinate-readout-bg {{ fill: #ffffff; fill-opacity: 0.92; stroke: #d1d5db; stroke-width: 0.5; }}
    #coordinate-readout-text {{ fill: #111827; font: 11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  </style>
  <g id="coordinate-crosshair" clip-path="url(#coordinate-clip)">
    <line id="coordinate-vline" x1="{left:.6f}" y1="{top:.6f}" x2="{left:.6f}" y2="{bottom:.6f}"/>
    <line id="coordinate-hline" x1="{left:.6f}" y1="{top:.6f}" x2="{right:.6f}" y2="{top:.6f}"/>
  </g>
  <g id="coordinate-readout">
    <rect id="coordinate-readout-bg" x="{readout_x:.6f}" y="{readout_y:.6f}" width="252" height="22" rx="4"/>
    <text id="coordinate-readout-text" x="{readout_x + 6:.6f}" y="{text_y:.6f}">Move mouse over plot</text>
  </g>
  <rect id="coordinate-hit-area" x="{left:.6f}" y="{top:.6f}" width="{width:.6f}" height="{height:.6f}" fill="transparent" pointer-events="all"/>
  <script type="application/ecmascript"><![CDATA[
(function () {{
  const metadata = {data_json};
  const svg = document.documentElement;
  const hitArea = document.getElementById("coordinate-hit-area");
  const crosshair = document.getElementById("coordinate-crosshair");
  const vline = document.getElementById("coordinate-vline");
  const hline = document.getElementById("coordinate-hline");
  const readoutBg = document.getElementById("coordinate-readout-bg");
  const readoutText = document.getElementById("coordinate-readout-text");
  const svgPoint = svg.createSVGPoint();

  function eventPoint(event) {{
    const screenMatrix = svg.getScreenCTM();
    if (!screenMatrix) {{
      return null;
    }}
    svgPoint.x = event.clientX;
    svgPoint.y = event.clientY;
    return svgPoint.matrixTransform(screenMatrix.inverse());
  }}

  function setReadout(text) {{
    readoutText.textContent = text;
    const bbox = readoutText.getBBox();
    readoutBg.setAttribute("width", Math.ceil(bbox.width + 12));
  }}

  function formatCycles(cycles) {{
    return Math.round(cycles).toLocaleString() + " cycles (" + (cycles / 1e9).toFixed(3) + "B)";
  }}

  function showCrosshair(point) {{
    crosshair.style.display = "inline";
    vline.setAttribute("x1", point.x);
    vline.setAttribute("x2", point.x);
    hline.setAttribute("y1", point.y);
    hline.setAttribute("y2", point.y);
  }}

  function hideCrosshair() {{
    crosshair.style.display = "none";
  }}

  hitArea.addEventListener("mousemove", function (event) {{
    const point = eventPoint(event);
    if (!point) {{
      return;
    }}

    const axes = metadata.axes;
    const xFraction = (point.x - axes.left) / (axes.right - axes.left);
    const yFraction = (point.y - axes.top) / (axes.bottom - axes.top);
    const stride = metadata.xlim[0] + xFraction * (metadata.xlim[1] - metadata.xlim[0]);
    const cycles = metadata.ylim[1] - yFraction * (metadata.ylim[1] - metadata.ylim[0]);

    showCrosshair(point);
    setReadout("stride=" + stride.toFixed(2) + "  cycles=" + formatCycles(cycles));
  }});

  hitArea.addEventListener("mouseleave", function () {{
    hideCrosshair();
    setReadout("Move mouse over plot");
  }});

  setReadout("Move mouse over plot");
}})();
  ]]></script>
</g>"""


def main():
    parser = argparse.ArgumentParser(
        description="Plot median cycle counts by stride from benchmark output."
    )
    parser.add_argument("input", nargs="?", type=Path, default=Path("result.txt"))
    parser.add_argument("output", nargs="?", type=Path, default=Path("graph.svg"))
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="number of runs per stride required before taking the median",
    )
    args = parser.parse_args()

    runs_by_stride, ignored = parse_runs(args.input)
    median_cycles, incomplete = median_cycles_by_stride(runs_by_stride, args.runs)

    if not median_cycles:
        raise SystemExit(f"No strides with at least {args.runs} runs found in {args.input}")

    interactive = plot_median_cycles(median_cycles, args.output)

    print(f"Wrote {args.output} using {len(median_cycles)} complete stride groups.")
    if interactive:
        print("Open the SVG directly in a browser for mouse coordinate readout.")
    else:
        print("Mouse coordinate readout is only embedded when the output path ends in .svg.")
    if incomplete:
        skipped = ", ".join(
            f"stride {stride} ({count}/{args.runs} runs)"
            for stride, count in incomplete.items()
        )
        print(f"Skipped incomplete groups: {skipped}")
    if ignored:
        print(f"Ignored {ignored} unparsable line(s).")


if __name__ == "__main__":
    main()

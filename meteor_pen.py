#!/usr/bin/env python3
"""
Meteor pen — draw fake meteor streaks over the stream to test the detector.

A floating, transparent, always-on-top pad. Click-drag to draw a bright
streak that fades after ~1.4 s. Draw over the telescope video: the detector
sees the stroke as a meteor and triggers the chime/flash/event. Strokes also
appear on the stream if it captures the desktop (e.g. OBS screen or window
capture that includes this pad).

The pad is a borderless, non-activating panel that never steals focus from
the video. Drag the dark strip at the top to move it; click the × in the
top-right corner to quit; Ctrl+C in the terminal also quits.

Usage:
  python meteor_pen.py                        # default 520x340 pad at top-left
  python meteor_pen.py --x 200 --y 150 --w 800 --h 500
"""

import argparse
import time

import objc
from AppKit import (
    NSApplication, NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered, NSBezierPath, NSColor, NSFloatingWindowLevel,
    NSFont, NSFontAttributeName, NSNonactivatingPanelMask, NSPanel,
    NSRectFill, NSString, NSTimer, NSView, NSWindowStyleMaskBorderless,
    NSEvent, NSMakeRect,
)

BRIGHT = (1.00, 0.985, 0.90)
DIM = (0.56, 0.54, 0.40)
FADE_MS = 1.4
GONE_MS = 1.9
HANDLE_H = 26


class PenView(NSView):
    def init(self):
        self = objc.super(PenView, self).init()
        if self is None:
            return None
        self.strokes = []   # list of (points, created_timestamp)
        self.cur = []       # points of the stroke being drawn
        self.dragWin = None  # (frame_origin, global_mouse_x, global_mouse_y)
        return self

    def isFlipped(self):
        return True

    def mouseDown_(self, e):
        p = self.convertPoint_fromView_(e.locationInWindow(), None)
        w = self.bounds().size.width
        if p.x > w - 28 and p.y < 28:      # close button
            NSApplication.sharedApplication().terminate_(None)
            return
        if p.y < HANDLE_H:                  # grab the handle to move the pad
            origin = self.window().frame().origin
            g = NSEvent.mouseLocation()
            self.dragWin = (origin, g.x, g.y)
            return
        self.cur = [p]

    def mouseDragged_(self, e):
        p = self.convertPoint_fromView_(e.locationInWindow(), None)
        if self.dragWin is not None:
            origin, gx, gy = self.dragWin
            g = NSEvent.mouseLocation()
            self.window().setFrameOrigin_((origin.x + g.x - gx,
                                           origin.y + g.y - gy))
            return
        if self.cur is not None:
            self.cur.append(p)
            self.setNeedsDisplay_(True)

    def mouseUp_(self, _e):
        if self.dragWin is not None:
            self.dragWin = None
            return
        if self.cur:
            if len(self.cur) > 1:
                self.strokes.append((list(self.cur), time.time()))
            self.cur = []
            self.setNeedsDisplay_(True)

    def tick_(self, _timer):
        now = time.time()
        self.strokes = [s for s in self.strokes if now - s[1] < GONE_MS]
        self.setNeedsDisplay_(True)

    def _stroke_path(self, points):
        path = NSBezierPath.bezierPath()
        path.setLineWidth_(9.0)
        path.setLineCapStyle_(1)   # round
        path.setLineJoinStyle_(1)  # round
        path.moveToPoint_(points[0])
        for p in points[1:]:
            path.lineToPoint_(p)
        return path

    def drawRect_(self, rect):
        try:
            NSColor.clearColor().set()
            NSRectFill(self.bounds())
            b = self.bounds()
            w, h = b.size.width, b.size.height
            now = time.time()

            for pts, t0 in self.strokes:
                age = now - t0
                if age > FADE_MS:
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(*DIM, 1).set()
                else:
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(*BRIGHT, 1).set()
                self._stroke_path(pts).stroke()

            if len(self.cur) > 1:
                NSColor.colorWithCalibratedRed_green_blue_alpha_(*BRIGHT, 1).set()
                self._stroke_path(self.cur).stroke()

            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.11, 0.15, 0.28, 1).set()
            NSBezierPath.bezierPathWithRect_(NSMakeRect(0, 0, w, HANDLE_H)).fill()
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.23, 0.27, 0.38, 1).set()
            NSBezierPath.bezierPathWithRect_(NSMakeRect(0.5, 0.5, w - 1, h - 1)).stroke()

            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.56, 0.63, 0.79, 1).set()
            NSString.alloc().initWithString_(
                "METEOR PEN — drag to draw · top strip moves · × quits"
            ).drawAtPoint_withAttributes_(
                (8, 8),
                {NSFontAttributeName: NSFont.systemFontOfSize_(11)},
            )

            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.55, 0.18, 0.22, 1).set()
            NSBezierPath.bezierPathWithRect_(NSMakeRect(w - 20, 4, 14, 14)).stroke()
        except Exception:
            import traceback
            traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", type=int, default=0)
    ap.add_argument("--y", type=int, default=200)
    ap.add_argument("--w", type=int, default=520)
    ap.add_argument("--h", type=int, default=340)
    args = ap.parse_args()

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    mask = NSWindowStyleMaskBorderless | NSNonactivatingPanelMask
    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(args.x, args.y, args.w, args.h), mask,
        NSBackingStoreBuffered, False)
    view = PenView.alloc().init()
    panel.setContentView_(view)
    panel.setOpaque_(False)
    panel.setBackgroundColor_(NSColor.clearColor())
    panel.setLevel_(NSFloatingWindowLevel)
    panel.setBecomesKeyOnlyIfNeeded_(True)
    panel.setIgnoresMouseEvents_(False)
    panel.orderFrontRegardless()

    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.1, view, objc.selector(view.tick_), None, True)

    print("[pen] ready — drag to draw a meteor; top strip moves; × or Ctrl+C quits")
    app.run()


if __name__ == "__main__":
    main()

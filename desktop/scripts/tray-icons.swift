import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

// Menu bar template images (black, alpha only) of the gaze mark: awake (rings and pupil) and
// asleep (the bottom of each ring, eyes closed). 54x44 px: macOS draws them 18pt tall.
//   swift scripts/tray-icons.swift src-tauri/icons
let (w, h) = (54, 44)
let scale: CGFloat = 1.78
func render(_ asleep: Bool, _ path: String) {
  let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
  // SVG coordinates (y down, the mark centred on 16,16) into the image.
  ctx.translateBy(x: CGFloat(w) / 2, y: CGFloat(h) / 2)
  ctx.scaleBy(x: scale, y: -scale)
  ctx.translateBy(x: -16, y: -16)
  ctx.setStrokeColor(CGColor(gray: 0, alpha: 1))
  ctx.setFillColor(CGColor(gray: 0, alpha: 1))
  ctx.setLineWidth(3.2)
  ctx.setLineCap(.round)
  for cx in [9.5, 22.5] {
    if asleep {
      // A closed lid: the bottom of the ring, apart from its neighbour, on the eyes' middle line.
      ctx.addArc(center: CGPoint(x: cx, y: 10), radius: 6.5, startAngle: 0.9, endAngle: .pi - 0.9, clockwise: false)
      ctx.strokePath()
    } else {
      ctx.strokeEllipse(in: CGRect(x: cx - 6.5, y: 9.5, width: 13, height: 13))
    }
  }
  if !asleep { ctx.fillEllipse(in: CGRect(x: 22.5 - 2.6, y: 16 - 2.6, width: 5.2, height: 5.2)) }
  let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
  CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
  CGImageDestinationFinalize(dest)
}
let dir = CommandLine.arguments[1]
render(false, "\(dir)/tray-awake.png")
render(true, "\(dir)/tray-asleep.png")

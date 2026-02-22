import fitz

def main():
    doc = fitz.open()
    page = doc.new_page(width=500, height=200)
    
    text = "System 2 Thinking"
    
    x0, y0, x1, y1 = 50, 50, 450, 100
    box_w = x1 - x0
    box_h = y1 - y0
    
    tw = fitz.TextWriter(page.rect)
    font = fitz.Font("cjk")
    fontsize = box_h * 0.8
    
    tl = font.text_length(text, fontsize=fontsize)
    tl = max(tl, 1)
    
    ratio = box_w / tl
    
    pt = fitz.Point(x0, y1 - box_h * 0.2)
    
    tw.append(pt, text, font=font, fontsize=fontsize)
    
    mat = fitz.Matrix(ratio, 0, 0, 1, 0, 0)
    tw.write_text(page, render_mode=0, morph=(pt, mat)) # render_mode=0 to verify visually
    
    # Draw bbox for reference
    page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(1,0,0), width=1)
    
    doc.save("test_stretch.pdf")
    print("test_stretch.pdf saved")

if __name__ == "__main__":
    main()

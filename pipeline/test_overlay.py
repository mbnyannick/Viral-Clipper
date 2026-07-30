from PIL import Image, ImageDraw, ImageFont

CANVAS_W = 720
MAX_LINE_WIDTH = 580
PADDING_TOP = 15
PADDING_BOTTOM = 15
LINE_GAP = 8
WORD_GAP = 10

def draw_test():
    lines = [
        "She texted him mid-stream",
        "\"You're under NDA or I'll SUE\"",
        "N3on: \"I have great lawyers. Let's go.\""
    ]
    
    # Try to load Arial or some default font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 36)
    except:
        font = ImageFont.load_default(36)
        
    line_dims = []
    for line in lines:
        bbox = font.getbbox(line)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        line_dims.append((w, h))
        
    text_total_h = sum(h for _, h in line_dims) + LINE_GAP * (len(line_dims) - 1)
    total_h = PADDING_TOP + text_total_h + PADDING_BOTTOM
    
    img = Image.new("RGBA", (CANVAS_W, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    y = PADDING_TOP
    bg_pad_x = 20
    bg_pad_y = 10
    
    for (line_w, line_h), line in zip(line_dims, lines):
        x = (CANVAS_W - line_w) // 2
        
        draw.rounded_rectangle(
            [(x - bg_pad_x, y - bg_pad_y), (x + line_w + bg_pad_x, y + line_h + bg_pad_y)],
            radius=16,
            fill=(255, 255, 255)
        )
        
        draw.text((x, y), line, font=font, fill=(0, 0, 0))
        y += line_h + LINE_GAP
        
    img.save("test_out.png")
    print("Saved test_out.png")

if __name__ == "__main__":
    draw_test()

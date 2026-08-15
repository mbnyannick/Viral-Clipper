import json
import urllib.request
import urllib.error
import ssl

ZERNIO_KEY = "sk_02b8f460ab56bace734931e41ab0c7ae01a737dbf20121dd870cc899f42d586d"
ZERNIO_YT_KEY = "sk_0a3c312652a472d2595d46e5ea7a528ac089fc77293a2dbee66324f51deec245"
WOOP_IG_KEY = "wsk_70ad62bb887c1784.acb8bf93dff7420bf33430b5e0ed5057ceea80d43074526a854cf96b99528804"
WOOP_FB_KEY = "wsk_6fc4363d65332255.bbbf5403a2c42f58c0e2f5cebdf77e04d9e90a0b88dd19d8639299a90c6f1d39"

def post_zernio(platform: str, title: str, content: str, video_url: str):
    account_id = "6a7126f0eb10586dadc92b3a" if platform == "tiktok" else "6a7126d9eb10586dadc9277a"
    api_key = ZERNIO_KEY if platform == "tiktok" else ZERNIO_YT_KEY
    payload = {
        "platforms": [{"platform": platform, "accountId": account_id}],
        "title": title[:95],
        "content": content,
        "mediaItems": [{"type": "video", "url": video_url}],
        "publishNow": True
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.zernio.com/v1/posts",
        data=data_bytes,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ViralBot/1.0"
        },
        method="POST"
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=15.0, context=ctx) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as he:
        return he.code, he.read().decode("utf-8")

def post_woopsocial(platform: str, title: str, content: str, video_url: str):
    is_ig = (platform == "instagram")
    api_key = WOOP_IG_KEY if is_ig else WOOP_FB_KEY
    project_id = "157994660491427840" if is_ig else "157993243961720832"
    social_account_id = "157994846563336192" if is_ig else "158008660860076032"
    plat_str = "INSTAGRAM" if is_ig else "FACEBOOK"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # 1. Download video binary
    print(f"Downloading video for {plat_str} upload...")
    req_v = urllib.request.Request(video_url, headers={"User-Agent": "ViralBot/1.0"})
    with urllib.request.urlopen(req_v, timeout=30.0, context=ctx) as v_resp:
        video_bytes = v_resp.read()

    # 2. Upload media to WoopSocial
    boundary = "----ViralBoundary123456789"
    body_parts = []
    body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
    body_parts.append(b'Content-Disposition: form-data; name="file"; filename="clip.mp4"\r\n')
    body_parts.append(b"Content-Type: video/mp4\r\n\r\n")
    body_parts.append(video_bytes)
    body_parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    form_data = b"".join(body_parts)

    req_upload = urllib.request.Request(
        f"https://api.woopsocial.com/v1/media?projectId={project_id}",
        data=form_data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "ViralBot/1.0"
        },
        method="POST"
    )
    with urllib.request.urlopen(req_upload, timeout=60.0, context=ctx) as u_resp:
        u_data = json.loads(u_resp.read().decode("utf-8"))
        media_id = u_data.get("mediaId") or u_data.get("id")

    print(f"Uploaded media to WoopSocial {plat_str}: mediaId={media_id}")

    # 3. Create post
    post_payload = {
        "content": [
            {
                "text": content,
                "media": [{"type": "MEDIA_LIBRARY", "mediaId": media_id}]
            }
        ],
        "schedule": {"type": "PUBLISH_NOW"},
        "socialAccounts": [
            {
                "platform": plat_str,
                "socialAccountId": social_account_id,
                "postType": "REEL"
            }
        ]
    }
    p_bytes = json.dumps(post_payload).encode("utf-8")
    req_post = urllib.request.Request(
        "https://api.woopsocial.com/v1/posts",
        data=p_bytes,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ViralBot/1.0"
        },
        method="POST"
    )
    with urllib.request.urlopen(req_post, timeout=15.0, context=ctx) as p_resp:
        return p_resp.status, p_resp.read().decode("utf-8")

print("Testing WoopSocial Instagram...")
status, body = post_woopsocial("instagram", "Test Title", "Test Caption #Reels", "https://gadgets-semiconductor-icons-oriental.trycloudflare.com/clips/clip_001.mp4")
print(f"WoopSocial IG Response ({status}): {body[:300]}")

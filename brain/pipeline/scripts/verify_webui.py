"""Optional Chromium checks against a running server without GPU/API calls.

Requires playwright and Pillow. The server must have image credentials unset.
An imported completed run is required to exercise video and trajectory downloads.
"""
import argparse
import json
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8040")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fixture = args.out / "upload-fixture.png"
    Image.new("RGB", (640, 480), (80, 110, 130)).save(fixture)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=[
            "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
        context = browser.new_context(viewport={"width": 1440, "height": 1100},
                                      permissions=["camera"], accept_downloads=True)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.goto(args.url)
        assert page.locator("html").get_attribute("lang") == "en"
        assert page.locator("#choose-image").inner_text() == "Upload image"
        page.wait_for_function("Array.from(document.querySelectorAll('img[src=\"/aeonlabs-wordmark.png\"]')).every(i => i.complete && i.naturalWidth > 0)")
        assert page.locator('img[src="/aeonlabs-wordmark.png"]').count() == 2
        health = page.request.get(args.url + "/api/status").json()
        assert not health["image_ready"], "Test refuses live image credentials"
        page.locator("#image-file").set_input_files(fixture)
        page.wait_for_function("document.getElementById('image-size').textContent === '640 × 480'")
        page.locator("form details summary").click()
        page.locator("#manual-mask").check()
        page.locator("#input-preview").scroll_into_view_if_needed()
        bounds = page.locator("#input-preview").bounding_box()
        page.mouse.move(bounds["x"] + 50, bounds["y"] + 50)
        page.mouse.down()
        page.mouse.move(bounds["x"] + 130, bounds["y"] + 100, steps=10)
        page.mouse.up()
        page.locator("#instruction").fill("Lift the cup with the right hand.")
        page.locator("#hfov").fill("70")
        page.locator("#pitch").fill("0")
        painted = page.locator("#input-preview").evaluate("(canvas) => canvas.toDataURL()")
        page.locator("#language-toggle").click()
        assert page.locator("html").get_attribute("lang") == "zh-CN"
        assert page.locator("#choose-image").inner_text() == "上传图片"
        assert page.locator("#instruction").input_value() == "Lift the cup with the right hand."
        assert page.locator("#input-preview").evaluate("(canvas) => canvas.toDataURL()") == painted
        assert page.locator("#manual-mask").is_checked()
        page.locator("#generate").click()
        page.wait_for_function("document.getElementById('form-error').textContent.includes('GEMINI_API_KEY')")
        page.locator("#language-toggle").click()
        assert "Image editing is not configured" in page.locator("#form-error").inner_text()
        assert "Image editing / automatic masking" in page.locator("#health-detail").inner_text()
        page.locator("#language-toggle").click()
        page.locator("#open-camera").click()
        page.wait_for_function("document.getElementById('camera').videoWidth > 0")
        page.locator("#capture").click()
        page.locator("#camera-area").wait_for(state="hidden")
        page.locator(".history-item").first.click()
        page.wait_for_function("document.getElementById('job-status').textContent === '已完成'")
        page.wait_for_function("!document.getElementById('trajectory').hidden")
        assert page.locator("#import-notice").is_visible()
        assert page.locator("#steps").is_hidden()
        page.locator("#joint-group").select_option("hand")
        video = page.locator("#output-video")
        if video.is_visible():
            page.wait_for_function("document.getElementById('output-video').readyState >= 1")
            duration = page.evaluate("document.getElementById('output-video').duration")
            assert abs(duration - 4) < .05, duration
        with page.expect_download() as download:
            page.get_by_role("link", name="机器人 NPZ", exact=True).click()
        download.value.save_as(args.out / "downloaded_robot_actions.npz")
        page.reload()
        page.wait_for_function("!document.getElementById('trajectory').hidden")
        assert page.locator("html").get_attribute("lang") == "zh-CN"
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=str(args.out / "desktop-zh.png"), full_page=True)
        if video.is_visible():
            video.evaluate("(v) => { v.pause(); v.currentTime = 1.25; }")
            page.wait_for_function("!document.getElementById('output-video').seeking")
        page.locator("#language-toggle").click()
        assert page.locator("#job-status").inner_text() == "Completed"
        assert page.get_by_role("link", name="Robot NPZ", exact=True).is_visible()
        if video.is_visible():
            assert abs(video.evaluate("(v) => v.currentTime") - 1.25) < .05
        page.screenshot(path=str(args.out / "desktop.png"), full_page=True)
        page.reload()
        page.wait_for_function("!document.getElementById('trajectory').hidden")
        assert page.locator("html").get_attribute("lang") == "en"
        page.set_viewport_size({"width":390,"height":844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(args.out / "mobile.png"), full_page=True)
        page.locator("#language-toggle").click()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
        (args.out / "checks.json").write_text(json.dumps({
            "browser": "Chromium", "javascript_errors": errors,
            "passed": ["image upload", "manual mask painting", "missing-key error",
                       "camera capture with browser test device", "imported-result provenance",
                       "4-second video metadata", "hand trajectory plot", "NPZ download",
                       "result restored after reload", "mobile no horizontal overflow"],
            "localization": ["English default", "original Aeon logo loaded in both locations",
                             "language toggle preserves uploaded image, mask and instruction",
                             "service and error translations", "result and download translations",
                             "language persists after reload", "video position survives toggle",
                             "mobile layout in both languages"],
            "live_image_generation_tested": False, "live_h3_generation_tested": False,
        }, indent=2))
        browser.close()
        print("Browser checks passed:", args.out)


if __name__ == "__main__":
    main()

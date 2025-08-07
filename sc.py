import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By

# Setup Chrome driver (you can use Firefox or Edge too)
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
# options.add_argument("--headless")  # Optional: run without opitening a browser
driver = webdriver.Chrome(options=options)

# Replace with your actual URL
url = "https://www.neosofttech.com/blogs/"  # ⬅️ change this
driver.get(url)

# Wait for page to load completely (use WebDriverWait in production)
time.sleep(3)

# Target div with id="slick-slide01"
try:
    container = driver.find_element(By.ID, "slick-slide01")

    # Find all <img> tags inside the container
    img_tags = container.find_elements(By.TAG_NAME, "img")

    print(f"Found {len(img_tags)} image(s).")

    # Make folder to save images
    save_dir = "slick_slide_images"
    os.makedirs(save_dir, exist_ok=True)

    # Download each image
    for i, img in enumerate(img_tags):
        src = img.get_attribute("src")
        if not src:
            continue

        # Download image using requests
        try:
            response = requests.get(src, timeout=10)
            ext = src.split(".")[-1].split("?")[0]  # Get file extension
            filename = f"image_{i + 1}.{ext}"
            filepath = os.path.join(save_dir, filename)

            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Failed to download {src}: {e}")

except Exception as e:
    print("Could not find the slick-slide01 element:", e)

driver.quit()

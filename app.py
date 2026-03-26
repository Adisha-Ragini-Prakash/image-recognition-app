import streamlit as st
import os
import json
import re
import base64
import pandas as pd
from PIL import Image
import io
from groq import Groq

# ─────────────────────────────────────────────
# PASTE YOUR GROQ API KEY HERE
# ─────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

def extract_bill_data(image_bytes: bytes, bill_name: str) -> list[dict]:
    client = Groq(api_key=GROQ_API_KEY)

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = (img.format or "JPEG").lower()
        fmt = "jpeg" if fmt == "jpg" else fmt
    except Exception:
        fmt = "jpeg"
    media_type = f"image/{fmt}"

    system_prompt = """You are a precise data-entry specialist reading a bill or receipt image.

Your ONLY job: extract every printed row from the bill exactly as it appears.

OUTPUT FORMAT — return a raw JSON array, nothing else. No markdown, no ```json, no intro text.

Each object in the array:
{
  "item": "<exact name as printed on the bill>",
  "quantity": <number — read the Qty/Qty. column exactly>,
  "unit_price": <number — read the Price/Rate column exactly, no currency symbols>,
  "total_price": <number — read the Amount column exactly, no currency symbols>,
  "category": "<Food | Beverage | Tax | Discount | Service | Other | NetTotal | GrandTotal>"
}
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_b64}"}
                    },
                    {
                        "type": "text",
                        "text": "Read this bill image carefully. Extract every line item row exactly as printed. Return only the JSON array."
                    }
                ]
            }
        ],
        temperature=0.0,
        max_tokens=2048,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE).strip()
    raw = re.sub(r"```$", "", raw, flags=re.MULTILINE).strip()

    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                parsed = v
                break

    for item in parsed:
        item["source_bill"] = bill_name

    return parsed

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Hotel Bill Extractor", page_icon="🧾", layout="wide")

st.title("🧾 Hotel Bill Data Extractor")
st.markdown("Upload hotel or restaurant bill images. The app extracts every line item into a table exactly as printed.")

uploaded_files = st.file_uploader(
    "Upload bill images (JPG, PNG, WEBP, BMP)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} file(s) ready.**")

    if st.button(" Extract Data from Bills", type="primary"):

        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            st.error("❌ Set your Groq API key at the top of app.py or via: export GROQ_API_KEY=gsk_...")
            st.stop()

        all_items_combined = []
        all_tax_combined = []
        errors = []
        progress  = st.progress(0, text="Starting…")

        for idx, file in enumerate(uploaded_files):
            bill_name = file.name
            progress.progress(
                idx / len(uploaded_files),
                text=f"Processing {bill_name} ({idx + 1} of {len(uploaded_files)})…"
            )

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(Image.open(file), caption=bill_name, width=300)

            with col2:
                with st.spinner(f"Reading {bill_name}…"):
                    try:
                        file.seek(0)
                        items = extract_bill_data(file.read(), bill_name)

                        bill_items = []
                        bill_tax = []

                        for it in items[:]:
                            if it['category'] == 'Tax' or re.search(r'CGST|SGST|VAT', it['item'], re.I):
                                bill_tax.append(it)
                                items.remove(it)
                            elif it['category'] in ['NetTotal', 'GrandTotal']:
                                items.remove(it)
                            else:
                                bill_items.append(it)

                        all_items_combined.extend(bill_items)
                        all_tax_combined.extend(bill_tax)

                        if bill_items:
                            st.subheader(f"📄 Items from {bill_name}")
                            df = pd.DataFrame(bill_items)
                            st.dataframe(df, use_container_width=True, hide_index=True)

                        if bill_tax:
                            st.subheader(f"💰 Tax Details from {bill_name}")
                            tax_df = pd.DataFrame(bill_tax)
                            st.dataframe(tax_df, use_container_width=True, hide_index=True)

                    except json.JSONDecodeError:
                        msg = f"Groq returned non-JSON for {bill_name}. Try uploading a clearer image."
                        st.error(f"❌ {msg}")
                        errors.append(msg)

                    except Exception as e:
                        err_str = str(e)
                        if "invalid_api_key" in err_str or "401" in err_str:
                            st.error("❌ Invalid Groq API key. Check your key at console.groq.com")
                            st.stop()
                        elif "rate_limit" in err_str or "429" in err_str:
                            st.warning(f"⚠️ Rate limit hit on {bill_name}. Wait a moment and retry.")
                            errors.append(f"{bill_name}: rate limit")
                        else:
                            st.error(f"❌ Error on {bill_name}: {e}")
                            errors.append(f"{bill_name}: {e}")

        progress.progress(1.0, text="All bills processed!")

        if all_items_combined:
            st.divider()
            st.subheader(" Combined Table — All Bills")
            combined_df = pd.DataFrame(all_items_combined)
            st.dataframe(combined_df, use_container_width=True, hide_index=True)

        if all_tax_combined:
            st.subheader("💰 Combined Tax Details — All Bills")
            combined_tax_df = pd.DataFrame(all_tax_combined)
            st.dataframe(combined_tax_df, use_container_width=True, hide_index=True)

        if errors:
            st.divider()
            for err in errors:
                st.warning(f"⚠️ {err}")

else:
    st.info("👆 Upload one or more bill images above to get started.")

st.divider()
st.caption("Built with Streamlit · Groq LLaMA 4 Scout Vision · Python")
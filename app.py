"""
Phần Mềm Chuyển Ảnh Thành Tranh Vẽ (Creative Image-to-Art Studio)
Thuần NumPy - Tuyệt đối không dùng OpenCV.
Hỗ trợ song ngữ Tiếng Việt thuần túy và Tiếng Anh (Song ngữ Anh - Việt).
"""

import base64
import io
import time
from pathlib import Path

import numpy as np
import pydicom
import streamlit as st
from PIL import Image

from src.cartoon import cartoonify
from src.color_adjust import apply_tone_adjustments
from src.edges import compute_gradients, continuous_line_art, create_edge_mask
from src.filters import gaussian_blur
from src.grayscale import to_grayscale
from src.io_handler import to_float32, to_uint8
from src.sketch import color_pencil_sketch, pencil_sketch
from src.watercolor import watercolor_effect

LOGO_FILE = Path("assets/logo.png")

# Từ điển bản dịch song ngữ Anh - Việt
TRANSLATIONS = {
    "vi": {
        "app_title": "Chuyển Ảnh Thành Tranh Vẽ",
        "brand_title": "Xưởng Nghệ Thuật",
        "lang_selector_label": "🌐 Ngôn ngữ / Language",
        "sidebar_style_header": "Chọn phong cách tranh",
        "style_label": "Phong cách vẽ:",
        "style_cartoon": "Tranh hoạt hình",
        "style_color_pencil": "Tranh chì màu",
        "style_pencil": "Tranh phác thảo chì",
        "style_watercolor": "Tranh màu nước",
        "style_edge": "Nét vẽ liền",
        "param_header": "Tinh chỉnh nét vẽ & màu sắc",
        "num_levels": "Số mảng màu:",
        "num_levels_help": "Số lượng mảng màu sắc. Giá trị thấp tạo mảng màu phẳng rõ rệt hơn.",
        "edge_thresh_cartoon": "Độ đậm nét viền mực:",
        "edge_thresh_cartoon_help": "Giá trị càng nhỏ thì nét viền mực càng đậm và rõ ràng.",
        "bilat_d": "Độ mịn mảng màu:",
        "bilat_sigma_s": "Độ loang không gian:",
        "bilat_sigma_r": "Độ mượt sắc độ:",
        "blur_sigma_pencil": "Độ đậm nét chì:",
        "blur_sigma_pencil_help": "Độ đậm nhạt của bóng chì than trên mặt giấy.",
        "blur_size_pencil": "Kích thước đầu bút chì:",
        "blur_size_pencil_help": "Độ dày của nét chì phác thảo.",
        "water_edge": "Độ đậm nét cọ viền:",
        "water_sat": "Độ rực rỡ màu nước:",
        "edge_thresh_sobel": "Ngưỡng nhận diện viền:",
        "edge_boldness": "Độ đậm nét mực:",
        "edge_boldness_help": "Tăng cường độ sâu mực đen thẳm, loại bỏ nét xám mờ nhạt.",
        "edge_thickness": "Độ dày nét vẽ:",
        "edge_thickness_help": "Độ dày pixel của nét vẽ (1: Mảnh tinh tế, 2: Vừa vặn, 3: Đậm nét).",
        "connect_lines": "Nối liền nét vẽ đứt đoạn",
        "connect_lines_help": "Tự động kết nối các khoảng hở 1-2 pixel để tạo nét vẽ liền mạch không bị chấm vụn.",
        "smooth_edge": "Nét vẽ khử răng cưa mượt mà",
        "invert_mask": "Nền trắng nét đen",
        "tone_header": "Độ sáng & Tương phản",
        "brightness": "Độ sáng:",
        "contrast": "Độ tương phản:",
        "saturation": "Độ tươi màu:",
        "uploader_label": "Chọn hoặc kéo thả ảnh vào đây:",
        "no_image_title": "Chưa có ảnh nào được chọn",
        "no_image_desc": "Nhấn vào khung ở trên hoặc kéo thả ảnh của bạn vào để biến thành tranh vẽ ngay.",
        "original_title": "Ảnh gốc",
        "result_title": "Kết quả",
        "time_metric": "Thời gian xử lý: {:.1f}ms",
        "file_info": "Tệp: {} | {}x{} điểm ảnh | {} kênh",
        "download_button": "Tải ảnh về máy",
        "file_download_name": "tranh_ve_nghe_thuat.png"
    },
    "en": {
        "app_title": "Image to Art Converter",
        "brand_title": "Art Studio",
        "lang_selector_label": "🌐 Language / Ngôn ngữ",
        "sidebar_style_header": "Select Art Style",
        "style_label": "Style:",
        "style_cartoon": "Cartoon",
        "style_color_pencil": "Color Pencil Sketch",
        "style_pencil": "Pencil Sketch",
        "style_watercolor": "Watercolor",
        "style_edge": "Continuous Line Art",
        "param_header": "Stroke & Color Adjustments",
        "num_levels": "Color Levels:",
        "num_levels_help": "Number of color levels. Lower values create more distinct flat shading.",
        "edge_thresh_cartoon": "Ink Outline Strength:",
        "edge_thresh_cartoon_help": "Lower values produce bolder and darker ink outlines.",
        "bilat_d": "Color Smoothness:",
        "bilat_sigma_s": "Spatial Spread:",
        "bilat_sigma_r": "Color Sensitivity:",
        "blur_sigma_pencil": "Pencil Shading Depth:",
        "blur_sigma_pencil_help": "Intensity of graphite pencil shading on paper.",
        "blur_size_pencil": "Pencil Tip Size:",
        "blur_size_pencil_help": "Thickness of pencil sketch strokes.",
        "water_edge": "Brush Stroke Intensity:",
        "water_sat": "Watercolor Vibrancy:",
        "edge_thresh_sobel": "Edge Threshold:",
        "edge_boldness": "Ink Boldness / Density:",
        "edge_boldness_help": "Enhances deep black ink depth, eliminating faint and washed-out lines.",
        "edge_thickness": "Line Thickness:",
        "edge_thickness_help": "Stroke width in pixels (1: Fine, 2: Medium, 3: Bold).",
        "connect_lines": "Connect broken contours (Continuous Lines)",
        "connect_lines_help": "Automatically bridges 1-2 pixel gaps to create continuous unbroken strokes.",
        "smooth_edge": "Smooth Anti-Aliased Lines",
        "invert_mask": "White background black ink",
        "tone_header": "Brightness & Contrast",
        "brightness": "Brightness:",
        "contrast": "Contrast:",
        "saturation": "Color Vibrancy:",
        "uploader_label": "Choose or drop an image here:",
        "no_image_title": "No image selected",
        "no_image_desc": "Click the area above or drop your photo to transform it into artwork.",
        "original_title": "Original Image",
        "result_title": "Result",
        "time_metric": "Processing time: {:.1f}ms",
        "file_info": "File: {} | {}x{} px | {} channels",
        "download_button": "Download Image",
        "file_download_name": "artistic_painting.png"
    }
}

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Chuyển Ảnh Thành Tranh Vẽ",
    page_icon=str(LOGO_FILE) if LOGO_FILE.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_logo_base64() -> str:
    """Đọc ảnh logo chuyển thành chuỗi base64 để hiển thị trực tiếp trong HTML."""
    if LOGO_FILE.exists():
        with open(LOGO_FILE, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


# Custom CSS: Tone màu Xanh Biển Đại Dương, nút bấm rộng rãi, thân thiện
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header & Brand tone xanh biển */
    .app-header-container {
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, #0284C7, #06B6D4, #38BDF8, transparent) 1;
        margin-bottom: 1.5rem;
    }
    
    .app-logo {
        width: 60px;
        height: 60px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.45);
        object-fit: cover;
        border: 2px solid rgba(255, 255, 255, 0.95);
        transition: transform 0.3s ease;
    }
    .app-logo:hover {
        transform: rotate(-3deg) scale(1.05);
    }
    
    .app-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #0369A1 0%, #0284C7 35%, #0EA5E9 70%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding: 0;
    }
    
    /* Badge & Info */
    .metric-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 6px 16px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.12) 0%, rgba(6, 182, 212, 0.12) 100%);
        color: #0284C7;
        border: 1px solid rgba(2, 132, 199, 0.3);
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.12);
        margin-bottom: 0.8rem;
    }
    
    /* Empty State Box */
    .card-box {
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.04) 0%, rgba(6, 182, 212, 0.04) 100%);
        border: 1.5px dashed rgba(2, 132, 199, 0.35);
        border-radius: 14px;
        padding: 3.5rem 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .card-box:hover {
        border-color: #0284C7;
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.07) 0%, rgba(6, 182, 212, 0.07) 100%);
    }
    
    /* Section Titles với chấm xanh biển phát sáng */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .dot-origin {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        background: #0284C7;
        box-shadow: 0 0 10px rgba(2, 132, 199, 0.7);
        display: inline-block;
    }
    .dot-result {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0284C7, #06B6D4);
        box-shadow: 0 0 10px rgba(6, 182, 212, 0.8);
        display: inline-block;
    }
    
    /* Nút tải ảnh màu xanh biển đại dương rực rỡ */
    .stDownloadButton button {
        background: linear-gradient(135deg, #0369A1 0%, #0284C7 50%, #06B6D4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.7rem 1.5rem !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.4) !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
    }
    .stDownloadButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 26px rgba(6, 182, 212, 0.6) !important;
    }
    
    /* Sidebar Brand */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(2, 132, 199, 0.2);
        margin-bottom: 1rem;
    }
    .sidebar-logo {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(2, 132, 199, 0.35);
    }
    .sidebar-title {
        font-weight: 700;
        font-size: 1.05rem;
        background: linear-gradient(135deg, #0369A1, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)


def parse_uploaded_file(uploaded_file) -> tuple[np.ndarray, str]:
    """Nạp file ảnh người dùng tải lên, hỗ trợ cả ảnh thông thường và file DICOM .dcm."""
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    if filename.endswith(".dcm"):
        # Xử lý file y tế DICOM
        dcm_stream = io.BytesIO(file_bytes)
        ds = pydicom.dcmread(dcm_stream)
        pixel_array = ds.pixel_array.astype(np.float32)

        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        if slope != 1.0 or intercept != 0.0:
            pixel_array = pixel_array * float(slope) + float(intercept)

        if pixel_array.ndim == 3:
            pixel_array = pixel_array[0]

        img_float = to_float32(pixel_array)
        return img_float, uploaded_file.name
    else:
        # Xử lý ảnh thường qua PIL
        pil_img = Image.open(io.BytesIO(file_bytes))
        if pil_img.mode == "RGBA":
            rgb_img = Image.new("RGB", pil_img.size, (255, 255, 255))
            rgb_img.paste(pil_img, mask=pil_img.split()[3])
            arr = np.array(rgb_img)
        elif pil_img.mode in ("RGB", "L"):
            arr = np.array(pil_img)
        else:
            arr = np.array(pil_img.convert("RGB"))

        img_float = to_float32(arr)
        return img_float, uploaded_file.name


def main():
    logo_b64 = get_logo_base64()

    # Nút chuyển đổi ngôn ngữ Anh - Việt ở đỉnh sidebar
    lang_choice = st.sidebar.radio(
        "🌐 Ngôn ngữ / Language",
        ["Tiếng Việt", "English"],
        horizontal=True,
        key="app_language_selector"
    )
    lang = "vi" if lang_choice == "Tiếng Việt" else "en"
    t = TRANSLATIONS[lang]

    # Thanh bên Sidebar có logo và tên Studio
    if logo_b64:
        st.sidebar.markdown(f"""
        <div class="sidebar-brand">
            <img src="data:image/png;base64,{logo_b64}" class="sidebar-logo" alt="Logo" />
            <span class="sidebar-title">{t["brand_title"]}</span>
        </div>
        """, unsafe_allow_html=True)

    # Tiêu đề chính chỉ gồm logo và tên ứng dụng
    if logo_b64:
        st.markdown(f"""
        <div class="app-header-container">
            <img src="data:image/png;base64,{logo_b64}" class="app-logo" alt="Logo" />
            <h1 class="app-title">{t["app_title"]}</h1>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="app-header-container">
            <h1 class="app-title">{t["app_title"]}</h1>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar: Chọn phong cách tranh vẽ
    st.sidebar.markdown(f"### {t['sidebar_style_header']}")

    style_keys = [
        "style_cartoon",
        "style_color_pencil",
        "style_pencil",
        "style_watercolor",
        "style_edge"
    ]
    style_names = [t[k] for k in style_keys]

    selected_idx = st.sidebar.selectbox(
        t["style_label"],
        range(len(style_keys)),
        format_func=lambda i: style_names[i],
        key="style_mode_select"
    )
    current_style = style_keys[selected_idx]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {t['param_header']}")

    # Khởi tạo giá trị mặc định cho từng phong cách
    blur_sigma = 10.0
    blur_size = 21
    num_levels = 6
    edge_thresh = 0.10
    edge_boldness = 1.0
    edge_thickness = 1
    connect_lines = True
    bilat_d = 7
    bilat_sigma_s = 7.0
    bilat_sigma_r = 0.12
    water_edge_str = 0.45
    water_sat = 1.35
    smooth_edge = True
    invert_mask = True

    if current_style == "style_cartoon":
        num_levels = st.sidebar.slider(
            t["num_levels"],
            min_value=3,
            max_value=16,
            value=6,
            step=1,
            key="slider_cartoon_levels",
            help=t["num_levels_help"]
        )
        edge_thresh = st.sidebar.slider(
            t["edge_thresh_cartoon"],
            min_value=0.03,
            max_value=0.25,
            value=0.10,
            step=0.01,
            key="slider_cartoon_edge",
            help=t["edge_thresh_cartoon_help"]
        )
        bilat_d = st.sidebar.slider(t["bilat_d"], 3, 11, 7, step=2, key="slider_cartoon_d")
        bilat_sigma_s = st.sidebar.slider(t["bilat_sigma_s"], 1.0, 15.0, 7.0, step=0.5, key="slider_cartoon_sigma_s")
        bilat_sigma_r = st.sidebar.slider(t["bilat_sigma_r"], 0.02, 0.40, 0.12, step=0.01, key="slider_cartoon_sigma_r")
    elif current_style in ("style_color_pencil", "style_pencil"):
        blur_sigma = st.sidebar.slider(
            t["blur_sigma_pencil"],
            min_value=1.0,
            max_value=30.0,
            value=10.0,
            step=0.5,
            key="slider_pencil_sigma",
            help=t["blur_sigma_pencil_help"]
        )
        blur_size = st.sidebar.slider(
            t["blur_size_pencil"],
            min_value=5,
            max_value=45,
            value=21,
            step=2,
            key="slider_pencil_size",
            help=t["blur_size_pencil_help"]
        )
    elif current_style == "style_watercolor":
        water_edge_str = st.sidebar.slider(
            t["water_edge"],
            min_value=0.1,
            max_value=1.0,
            value=0.45,
            step=0.05,
            key="slider_water_edge"
        )
        water_sat = st.sidebar.slider(
            t["water_sat"],
            min_value=1.0,
            max_value=2.0,
            value=1.35,
            step=0.05,
            key="slider_water_sat"
        )
        bilat_d = st.sidebar.slider(t["bilat_d"], 3, 11, 7, step=2, key="slider_water_d")
    else:
        edge_thresh = st.sidebar.slider(
            t["edge_thresh_sobel"],
            min_value=0.03,
            max_value=0.35,
            value=0.10,
            step=0.01,
            key="slider_edge_thresh"
        )
        edge_boldness = st.sidebar.slider(
            t["edge_boldness"],
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.05,
            key="slider_edge_boldness",
            help=t["edge_boldness_help"]
        )
        edge_thickness = st.sidebar.slider(
            t["edge_thickness"],
            min_value=1,
            max_value=3,
            value=1,
            step=1,
            key="slider_edge_thickness",
            help=t["edge_thickness_help"]
        )
        connect_lines = st.sidebar.checkbox(
            t["connect_lines"],
            value=True,
            key="chk_connect_lines",
            help=t["connect_lines_help"]
        )
        smooth_edge = st.sidebar.checkbox(t["smooth_edge"], value=True, key="chk_smooth_edge")
        invert_mask = st.sidebar.checkbox(t["invert_mask"], value=True, key="chk_invert_mask")

    # Nhóm tinh chỉnh ánh sáng & độ tương phản
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {t['tone_header']}")
    bright_val = st.sidebar.slider(
        t["brightness"],
        min_value=-0.50,
        max_value=0.50,
        value=0.0,
        step=0.02,
        key="slider_brightness"
    )
    contrast_val = st.sidebar.slider(
        t["contrast"],
        min_value=0.20,
        max_value=2.50,
        value=1.0,
        step=0.05,
        key="slider_contrast"
    )
    sat_val = st.sidebar.slider(
        t["saturation"],
        min_value=0.00,
        max_value=2.50,
        value=1.0,
        step=0.05,
        key="slider_saturation"
    )

    # Khu vực tải ảnh chính diện với key cố định tránh mất trạng thái khi đổi ngôn ngữ
    uploaded = st.file_uploader(
        t["uploader_label"],
        type=["png", "jpg", "jpeg", "bmp", "dcm"],
        key="app_main_file_uploader"
    )

    # Lưu và quản lý ảnh trong session_state để chuyển đổi ngôn ngữ không bị mất ảnh
    if uploaded is not None:
        file_sig = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("cached_file_sig") != file_sig:
            parsed_img, parsed_name = parse_uploaded_file(uploaded)
            st.session_state["cached_img"] = parsed_img
            st.session_state["cached_fname"] = parsed_name
            st.session_state["cached_file_sig"] = file_sig
            st.session_state["cached_img_uint8"] = to_uint8(parsed_img)
            # Xóa các tầng cache cũ khi người dùng nạp ảnh mới
            st.session_state.pop("cached_raw_result", None)
            st.session_state.pop("cached_result_img", None)
            st.session_state.pop("cached_byte_im", None)
            st.session_state.pop("last_style_cache_key", None)
            st.session_state.pop("last_full_cache_key", None)
            st.session_state.pop("last_png_cache_key", None)
    else:
        # Nếu người dùng bấm xóa ảnh trên widget
        st.session_state.pop("cached_img", None)
        st.session_state.pop("cached_fname", None)
        st.session_state.pop("cached_file_sig", None)
        st.session_state.pop("cached_img_uint8", None)
        st.session_state.pop("cached_raw_result", None)
        st.session_state.pop("cached_result_img", None)
        st.session_state.pop("cached_byte_im", None)
        st.session_state.pop("last_style_cache_key", None)
        st.session_state.pop("last_full_cache_key", None)
        st.session_state.pop("last_png_cache_key", None)

    if "cached_img" not in st.session_state or st.session_state["cached_img"] is None:
        st.markdown(f"""
        <div class="card-box">
            <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; color: #0369A1;">{t["no_image_title"]}</div>
            <div style="font-size: 0.95rem; color: #64748B;">{t["no_image_desc"]}</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Nạp ảnh gốc 100% từ session_state
    current_img = st.session_state["cached_img"]
    filename = st.session_state["cached_fname"]
    orig_h, orig_w = current_img.shape[:2]
    num_channels = current_img.shape[2] if current_img.ndim == 3 else 1
    file_sig = st.session_state["cached_file_sig"]

    # 1. Trích xuất khóa định danh tham số phong cách vẽ
    if current_style == "style_color_pencil":
        style_params = (blur_size, blur_sigma)
    elif current_style == "style_pencil":
        style_params = (blur_size, blur_sigma)
    elif current_style == "style_cartoon":
        style_params = (num_levels, bilat_d, bilat_sigma_s, bilat_sigma_r, edge_thresh)
    elif current_style == "style_watercolor":
        style_params = (water_edge_str, water_sat, bilat_d)
    else:
        style_params = (edge_thresh, edge_boldness, edge_thickness, connect_lines, smooth_edge, invert_mask)

    style_cache_key = (file_sig, current_style, style_params)

    # 2. Bộ nhớ đệm Tầng 1 (Style Filter Cache):
    # Nếu phong cách và thông số vẽ không đổi (ví dụ: chuyển ngôn ngữ, chuyển tab hoặc chỉnh độ sáng),
    # dùng ngay kết quả đã tính toán mà không cần chạy lại bộ lọc nặng (0ms tức thì).
    t_start = time.perf_counter()
    if (
        st.session_state.get("last_style_cache_key") == style_cache_key
        and "cached_raw_result" in st.session_state
    ):
        raw_result = st.session_state["cached_raw_result"]
        render_time_ms = st.session_state.get("cached_render_time_ms", 0.0)
    else:
        if current_style == "style_color_pencil":
            raw_result = color_pencil_sketch(current_img, blur_size=blur_size, blur_sigma=blur_sigma, as_float=True)
        elif current_style == "style_pencil":
            raw_result = pencil_sketch(current_img, blur_size=blur_size, blur_sigma=blur_sigma, as_float=True)
        elif current_style == "style_cartoon":
            raw_result = cartoonify(
                current_img,
                num_levels=num_levels,
                d=bilat_d,
                sigma_s=bilat_sigma_s,
                sigma_r=bilat_sigma_r,
                edge_threshold=edge_thresh,
                as_float=True
            )
        elif current_style == "style_watercolor":
            raw_result = watercolor_effect(
                current_img,
                bilat_d=bilat_d,
                edge_strength=water_edge_str,
                saturation_boost=water_sat,
                as_float=True
            )
        else:
            raw_result = continuous_line_art(
                current_img,
                threshold=edge_thresh,
                boldness=edge_boldness,
                thickness=edge_thickness,
                connect_lines=connect_lines,
                smooth=smooth_edge,
                invert=invert_mask,
                as_float=True
            )

        render_time_ms = (time.perf_counter() - t_start) * 1000
        st.session_state["cached_raw_result"] = raw_result
        st.session_state["last_style_cache_key"] = style_cache_key
        st.session_state["cached_render_time_ms"] = render_time_ms

    # 3. Bộ nhớ đệm Tầng 2 (Tone Adjustments Cache):
    tone_params = (bright_val, contrast_val, sat_val)
    full_cache_key = (style_cache_key, tone_params)

    if (
        st.session_state.get("last_full_cache_key") == full_cache_key
        and "cached_result_img" in st.session_state
    ):
        result_img = st.session_state["cached_result_img"]
    else:
        result_img = apply_tone_adjustments(
            raw_result,
            brightness=bright_val,
            contrast=contrast_val,
            saturation=sat_val
        )
        st.session_state["cached_result_img"] = result_img
        st.session_state["last_full_cache_key"] = full_cache_key

    # 4. Bộ nhớ đệm Tầng 3 (PNG Download Buffer Cache):
    # Loại bỏ triệt để việc nén lại file PNG 50MB (mất 1.7 giây) trên mỗi lần thao tác/chuyển màn hình
    if (
        st.session_state.get("last_png_cache_key") == full_cache_key
        and "cached_byte_im" in st.session_state
    ):
        byte_im = st.session_state["cached_byte_im"]
    else:
        out_uint8 = to_uint8(result_img)
        if out_uint8.ndim == 2:
            pil_export = Image.fromarray(out_uint8, mode="L")
        else:
            pil_export = Image.fromarray(out_uint8, mode="RGB")

        buf = io.BytesIO()
        pil_export.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.session_state["cached_byte_im"] = byte_im
        st.session_state["last_png_cache_key"] = full_cache_key

    # Layout song song 2 cột: Cột trái "Ảnh gốc", Cột phải "Kết quả"
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<div class="section-title"><span class="dot-origin"></span> {t["original_title"]}</div>', unsafe_allow_html=True)
        st.caption(t["file_info"].format(filename, orig_w, orig_h, num_channels))
        st.image(st.session_state.get("cached_img_uint8", to_uint8(current_img)), use_container_width=True)

    with col2:
        st.markdown(f'<div class="section-title"><span class="dot-result"></span> {t["result_title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-badge">{t["time_metric"].format(render_time_ms)}</div>', unsafe_allow_html=True)
        st.image(to_uint8(result_img), use_container_width=True)

        # Nút tải ảnh về máy
        st.download_button(
            label=t["download_button"],
            data=byte_im,
            file_name=t["file_download_name"],
            mime="image/png",
            use_container_width=True
        )


if __name__ == "__main__":
    main()

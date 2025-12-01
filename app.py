import streamlit as st
import tempfile
import os
import shutil
from backend import compare_students_deep, generate_visual_evidence

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Forensic Handwriting AI", 
    layout="wide", 
    page_icon="🕵️"
)

# Custom CSS for a professional academic look
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: 800; }
    .sub-text { font-size: 1.1rem; color: #4B5563; }
    .card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; border-left: 5px solid #3B82F6; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR (CONTROLS & DONATION)
# ==========================================
with st.sidebar:
    st.header("📁 Upload Evidence")
    st.info("Upload two images to detect handwriting similarity.")
    
    file1 = st.file_uploader("Document A (Suspect)", type=["jpg", "png", "jpeg"])
    file2 = st.file_uploader("Document B (Reference)", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    
    # --- VOLUNTARY DONATION (Optional) ---
    # st.header("☕ Support This Tool")
    # st.write("This tool is free for education. If it helped you, consider supporting the server costs.")
    # with st.expander("🇮🇶 Donate via ZainCash"):
    #     st.write("**ZainCash:** 0780 XXXXXXX")
    #     st.caption("Any amount helps!")
    
    st.markdown("---")
    st.caption("v1.0.0 | Built for Academic Integrity")

# ==========================================
# 3. MAIN APP LOGIC
# ==========================================
st.markdown('<p class="main-header">🕵️ Forensic Handwriting Analyst</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Deep Learning Identity Verification System</p>', unsafe_allow_html=True)

if file1 and file2:
    # Save files to temp directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as t1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as t2:
        t1.write(file1.getvalue())
        t2.write(file2.getvalue())
        path1, path2 = t1.name, t2.name

    # --- TABBED INTERFACE ---
    tab1, tab2 = st.tabs(["📊 Deep Metrics Analysis", "👁️ Visual Forensic Evidence"])

    # === TAB 1: THE METRICS ===
    with tab1:
        st.markdown("### 🤖 AI Feature Comparison")
        st.write("Using ResNet50 Deep Metric Learning to measure stylistic distance.")
        
        if st.button("🚀 Run Deep Analysis", type="primary"):
            with st.spinner("Analyzing stroke patterns..."):
                report = compare_students_deep(path1, path2)
                
                if report:
                    # Verdict Banner
                    if report['is_match']:
                        st.error(f"🚨 MATCH DETECTED ({report['confidence']:.1f}% Confidence)")
                        st.markdown(f"**Verdict:** These documents likely belong to the **same writer**.")
                    else:
                        st.success(f"✅ NO MATCH ({report['confidence']:.1f}% Confidence)")
                        st.markdown(f"**Verdict:** These documents likely belong to **different writers**.")
                    
                    # Metrics Grid
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Median Distance", f"{report['median_dist']:.3f}", help="Lower (< 0.6) is a match.")
                    c2.metric("Consistency Score", f"{report['consistency']:.1f}/100", help="How stable the handwriting is.")
                    c3.metric("Best Match", f"{report['best_match']:.3f}", help="The single most identical word pair.")
                    
                    # Histogram
                    st.bar_chart(report['raw_distances'])
                    st.caption("Distribution of Similarity (Left = Identical Strokes, Right = Different)")

    # === TAB 2: THE VISUAL PROOF ===
    with tab2:
        st.markdown("### 🔍 RANSAC Geometric Verification")
        st.write("This generates court-admissible visual proof by connecting mathematically identical stroke features.")
        
        # Direct Access - No Payment Wall!
        if st.button("Generate Visual Proof"):
            with st.spinner("Processing SIFT Features & Geometric RANSAC..."):
                count, img = generate_visual_evidence(path1, path2)
                
                if img is not None:
                    st.image(img, caption=f"Forensic Evidence: {count} Verified Matches", use_container_width=True)
                    st.success("Analysis Complete. Right-click the image to save.")
                else:
                    st.warning("Could not find enough distinct features to generate a clean visual graph.")

    # Cleanup (Optional)
    # os.remove(path1)
    # os.remove(path2)

else:
    # Empty State
    st.info("👈 Please upload two documents in the sidebar to begin.")
    
    # Demo Image (Optional: You can remove this)
    st.markdown("### How it works")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Deep Learning**")
        st.write("We extract 2,048 features from every word to find the writer's 'DNA'.")
    with c2:
        st.markdown("**2. Visual Geometry**")
        st.write("We verify matches using RANSAC to ignore noise and paper texture.")

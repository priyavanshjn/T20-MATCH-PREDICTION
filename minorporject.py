import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ======================================================
# LOAD MODEL
# ======================================================
def load_model():
    model_path = "t20_win_model.pkl"
    if os.path.exists(model_path):
        try:
            print("✅ Model loaded from file.")
            return joblib.load(model_path)
        except Exception as e:
            print("❌ Error loading model:", e)
    CTkMessagebox(title="Model Error", message="Model file not found or invalid!", icon="cancel")
    return None


# ======================================================
# DASHBOARD APP
# ======================================================
def launch_app(model):
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("🏏 T20 Win Predictor Dashboard")
    app.geometry("1000x700")
    app.resizable(False, False)

    # ================= HEADER =================
    header = ctk.CTkLabel(app,
                          text="🏏 T20 WIN PREDICTOR DASHBOARD",
                          font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"))
    header.pack(pady=15)

    # ================= MAIN FRAME =================
    frame_main = ctk.CTkFrame(app, corner_radius=15)
    frame_main.pack(padx=20, pady=10, fill="both", expand=True)

    frame_left = ctk.CTkFrame(frame_main, corner_radius=10)
    frame_left.pack(side="left", fill="y", padx=15, pady=15)

    frame_right = ctk.CTkFrame(frame_main, corner_radius=10)
    frame_right.pack(side="right", fill="both", expand=True, padx=15, pady=15)

    # ================= INPUT SECTION =================
    ctk.CTkLabel(frame_left, text="Enter Match Details",
                 font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold")).pack(pady=10)

    runs_var = ctk.StringVar()
    overs_var = ctk.StringVar()
    wickets_var = ctk.StringVar()
    target_var = ctk.StringVar()

    for label, var in [
        ("🏏 Runs Scored", runs_var),
        ("⏱ Overs Completed (e.g. 12.3)", overs_var),
        ("❌ Wickets Lost", wickets_var),
        ("🎯 Target Runs", target_var)
    ]:
        ctk.CTkLabel(frame_left, text=label).pack(anchor="w", padx=20, pady=(8, 2))
        ctk.CTkEntry(frame_left, textvariable=var, width=220).pack(padx=20, pady=(0, 8))

    # ================= PIE CHART SETUP =================
    fig, ax_pie = plt.subplots(figsize=(5, 4.5), dpi=100)
    fig.patch.set_facecolor("#2b2b2b")
    ax_pie.set_facecolor("#2b2b2b")

    wedges, texts, autotexts = ax_pie.pie([50, 50],
                                          labels=["Batting", "Bowling"],
                                          autopct="%1.1f%%",
                                          startangle=110,
                                          colors=["#00c853", "#ff5252"],
                                          textprops={"color": "white", "fontsize": 12})
    ax_pie.set_title("Win Probability", color="white", fontsize=14, pad=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Embed pie chart in the right frame
    canvas = FigureCanvasTkAgg(fig, master=frame_right)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)

    # ================= STATS DISPLAY AREA =================
    stats_frame = ctk.CTkFrame(frame_right, corner_radius=10)
    stats_frame.pack(pady=10, fill="x")

    stat_label = ctk.CTkLabel(stats_frame,
                              text="🏏 Stats will appear here after prediction",
                              font=ctk.CTkFont(family="Consolas", size=13),
                              justify="center")
    stat_label.pack(pady=10)

    # ================= PREDICT FUNCTION =================
    def show_error(msg):
        CTkMessagebox(title="⚠️ Input Error", message=msg, icon="warning")

    def update_pie(prob):
        ax_pie.clear()
        bowling_prob = 1 - prob
        values = [prob * 100, bowling_prob * 100]
        colors = ["#00e676", "#ff5252"]
        labels = [f"Batting ({values[0]:.1f}%)", f"Bowling ({values[1]:.1f}%)"]

        ax_pie.pie(values, labels=labels, autopct="%1.1f%%",
                   startangle=110, colors=colors,
                   textprops={"color": "white", "fontsize": 12})
        ax_pie.set_title("Win Probability", color="white", fontsize=14, pad=15)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        canvas.draw()

    def predict():
        try:
            runs = float(runs_var.get())
            overs = float(overs_var.get())
            wickets = float(wickets_var.get())
            target = float(target_var.get())
        except ValueError:
            show_error("Please enter valid numeric values.")
            return

        if overs <= 0 or overs > 20 or wickets < 0 or wickets > 10 or target <= 0:
            show_error("Please enter realistic match values.")
            return

        # Run Rates
        current_rr = runs / overs if overs > 0 else 0
        remaining_runs = target - runs
        remaining_overs = 20 - overs
        required_rr = (remaining_runs / remaining_overs) if remaining_overs > 0 else 0

        # Model Prediction
        try:
            X_new = pd.DataFrame([{"runs": runs, "wickets": wickets, "overs": overs, "target": target}])
            prob = model.predict_proba(X_new)[0][1]
        except Exception:
            prob = 0.5

        win_percent = round(prob * 100, 1)
        lose_percent = round(100 - win_percent, 1)

        # Update pie and stats
        progress_bar.set(win_percent / 100)
        update_pie(prob)

        stat_text = (
            f"🏏 Match Summary\n"
            f"──────────────────────────\n"
            f"Runs Scored     : {runs}\n"
            f"Wickets Lost    : {wickets}\n"
            f"Overs Completed : {overs}/20\n"
            f"Target Runs     : {target}\n\n"
            f"Batting Win %   : {win_percent}%\n"
            f"Bowling Win %   : {lose_percent}%\n\n"
            f"Current Run Rate: {current_rr:.2f}\n"
            f"Required Run Rate: {required_rr:.2f}\n"
            f"Projected Score : {runs / overs * 20:.0f} runs"
        )

        stat_label.configure(text=stat_text)

        # Commentary
        if prob > 0.8:
            commentary.configure(text="🔥 Batting team dominating!", text_color="#00e676")
        elif prob > 0.5:
            commentary.configure(text="⚖️ Match evenly poised.", text_color="#ffeb3b")
        else:
            commentary.configure(text="🎯 Bowling side in control.", text_color="#ff7043")

    # ================= BUTTONS & FOOTER =================
    ctk.CTkButton(frame_left, text="🚀 Predict Win Probability",
                  fg_color="#0078D4", hover_color="#005a9e",
                  font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                  command=predict).pack(pady=15)

    ctk.CTkLabel(frame_right, text="📊 Match Predictor",
                 font=ctk.CTkFont(family="Segoe UI", size=17, weight="bold")).pack(pady=(10, 5))

    progress_bar = ctk.CTkProgressBar(frame_right, width=400, height=20)
    progress_bar.set(0)
    progress_bar.pack(pady=10)

    commentary = ctk.CTkLabel(frame_right, text="Waiting for prediction...",
                              font=ctk.CTkFont(family="Segoe UI", size=12, slant="italic"),
                              wraplength=400)
    commentary.pack(pady=(10, 20))

    ctk.CTkLabel(app, text="© 2025 Priya | Minor ML Project",
                 font=ctk.CTkFont(family="Segoe UI", size=10, slant="italic")).pack(pady=10)

    # Safe Close
    app.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), app.destroy()))
    app.mainloop()


# ======================================================
# MAIN ENTRY POINT
# ======================================================
if __name__ == "__main__":
    model = load_model()
    if model:
        launch_app(model)

# app.py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib

# =============== FEATURE CONFIG ===============

FEATURES_INFO = [
    {
        "name": "experience_years",
        "label": "Experience (years)",
        "range": "0 – 40",
    },
    {
        "name": "test_score",
        "label": "Test Score",
        "range": "0 – 100",
    },
    {
        "name": "interview_score",
        "label": "Interview Score",
        "range": "0 – 10",
    },
    {
        "name": "communication",
        "label": "Communication Rating",
        "range": "0 – 10",
    },
]

REG_FEATURES = [f["name"] for f in FEATURES_INFO]
CLF_FEATURES = REG_FEATURES + ["performance_score"]

MODEL_PATHS = {
    "regression": "reg_model.pkl",
    "KNN": "knn_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "SVM": "svm_model.pkl",
    "Naive Bayes": "nb_model.pkl",
    "Logistic Regression": "log_model.pkl",
}

HIRE_LABELS = {0: "Reject", 1: "Hold", 2: "Hire"}

# =============== LOAD MODELS ===============

loaded_models = {}

try:
    loaded_models["regression"] = joblib.load(MODEL_PATHS["regression"])
except Exception as e:
    print("Error loading regression model:", e)

for key in ["KNN", "Decision Tree", "SVM", "Naive Bayes", "Logistic Regression"]:
    try:
        loaded_models[key] = joblib.load(MODEL_PATHS[key])
    except Exception as e:
        print(f"Error loading {key} model:", e)


# =============== GUI APP ===============

class SmartHiringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Hiring Assistant")
        self.root.configure(bg="#0f172a")

        # --- FULLSCREEN (Esc to exit) ---
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.exit_fullscreen)

        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Card.TFrame", background="#1f2937")
        style.configure("TLabel", background="#1f2937", foreground="#e5e7eb",
                        font=("SF Pro Text", 11))
        style.configure("Header.TLabel", background="#0f172a", foreground="#f9fafb",
                        font=("SF Pro Display", 22, "bold"))
        style.configure("SubHeader.TLabel", background="#0f172a", foreground="#9ca3af",
                        font=("SF Pro Text", 11))
        style.configure("Range.TLabel", background="#1f2937", foreground="#9ca3af",
                        font=("SF Pro Text", 10, "italic"))
        style.configure("TButton", font=("SF Pro Text", 11, "bold"), padding=10)
        style.map("TButton",
                  background=[("!disabled", "#3b82f6"), ("pressed", "#1d4ed8")],
                  foreground=[("!disabled", "#f9fafb")])
        style.configure("Result.TLabel", background="#111827", foreground="#f9fafb",
                        font=("SF Pro Display", 14, "bold"))

        outer = ttk.Frame(root, style="Card.TFrame", padding=24)
        outer.pack(expand=True, fill="both", padx=32, pady=24)

        header = ttk.Frame(outer, style="Card.TFrame")
        header.pack(fill="x")

        ttk.Label(header,
                  text="Automated Employee Evaluation Dashboard",
                  style="Header.TLabel").pack(anchor="w")

        ttk.Label(header,
                  text="Enter candidate details to predict performance score and hiring decision.",
                  style="SubHeader.TLabel").pack(anchor="w", pady=(4, 18))

        # ===== main split =====
        content = ttk.Frame(outer, style="Card.TFrame")
        content.pack(expand=True, fill="both")

        # ---------- Input card ----------
        input_card = ttk.Frame(content, style="Card.TFrame", padding=18)
        input_card.pack(side="left", fill="both", expand=True, padx=(0, 16), pady=(0, 4))

        ttk.Label(input_card,
                  text="Candidate Features",
                  font=("SF Pro Display", 15, "bold"),
                  background="#1f2937", foreground="#e5e7eb").grid(
                      row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        self.entries = []
        for i, info in enumerate(FEATURES_INFO, start=1):
            # label
            ttk.Label(input_card,
                      text=info["label"] + ":").grid(
                          row=i, column=0, sticky="w", pady=6, padx=(0, 12))

            # entry
            entry = ttk.Entry(input_card, width=18, font=("SF Pro Text", 11))
            entry.grid(row=i, column=1, sticky="ew", pady=6)
            self.entries.append(entry)

            # range hint
            ttk.Label(input_card,
                      text=f"Range: {info['range']}",
                      style="Range.TLabel").grid(
                          row=i, column=2, sticky="w", pady=6, padx=(12, 0))

        # Model dropdown
        ttk.Label(input_card,
                  text="Classification Model:",
                  padding=(0, 12, 0, 0)).grid(
                      row=len(FEATURES_INFO) + 1, column=0, sticky="w")

        self.model_var = tk.StringVar(value="Decision Tree")
        model_menu = ttk.Combobox(
            input_card,
            textvariable=self.model_var,
            values=["KNN", "Decision Tree", "SVM", "Naive Bayes", "Logistic Regression"],
            state="readonly",
        )
        model_menu.grid(row=len(FEATURES_INFO) + 1, column=1, sticky="ew", pady=(12, 0))

        # Buttons
        btn_frame = ttk.Frame(input_card, style="Card.TFrame")
        btn_frame.grid(row=len(FEATURES_INFO) + 2, column=0, columnspan=3,
                       pady=22, sticky="ew")

        ttk.Button(btn_frame, text="Evaluate Candidate",
                   command=self.evaluate_candidate).pack(side="left", padx=(0, 10))
        ttk.Button(btn_frame, text="Clear",
                   command=self.clear_inputs).pack(side="left")

        # ---------- Output card ----------
        output_card = ttk.Frame(content, style="Card.TFrame", padding=18)
        output_card.pack(side="right", fill="both", expand=True, padx=(16, 0), pady=(0, 4))

        ttk.Label(output_card,
                  text="Prediction Results",
                  font=("SF Pro Display", 15, "bold"),
                  background="#1f2937", foreground="#e5e7eb").pack(
                      anchor="w", pady=(0, 12))

        # performance box
        reg_box = ttk.Frame(output_card, style="Card.TFrame")
        reg_box.pack(fill="x", pady=(4, 14))

        ttk.Label(reg_box,
                  text="Predicted Performance Score",
                  background="#1f2937", foreground="#9ca3af",
                  font=("SF Pro Text", 11, "bold")).pack(anchor="w")

        self.reg_result = ttk.Label(reg_box, text="—", style="Result.TLabel", padding=10)
        self.reg_result.pack(fill="x", pady=(6, 0))

        # hire decision box
        clf_box = ttk.Frame(output_card, style="Card.TFrame")
        clf_box.pack(fill="x", pady=(14, 14))

        ttk.Label(clf_box,
                  text="Predicted Hiring Decision",
                  background="#1f2937", foreground="#9ca3af",
                  font=("SF Pro Text", 11, "bold")).pack(anchor="w")

        self.clf_result = ttk.Label(clf_box, text="—", style="Result.TLabel", padding=10)
        self.clf_result.pack(fill="x", pady=(6, 0))

        # small note showing that perf_score is used for classification
        ttk.Label(output_card,
                  text="Note: Hiring decision uses the predicted performance score along with other features.",
                  style="SubHeader.TLabel").pack(anchor="w", pady=(6, 0))

        self.status = ttk.Label(outer, text="Ready (press Esc to exit full screen)",
                                background="#1f2937",
                                foreground="#9ca3af", anchor="w", padding=(4, 4))
        self.status.pack(fill="x", pady=(16, 0))

    # ---------- helpers ----------

    def exit_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)

    def get_reg_features(self):
        values = []
        for i, entry in enumerate(self.entries):
            text = entry.get().strip()
            fname = REG_FEATURES[i]
            if text == "":
                raise ValueError(f"Please enter a value for '{fname}'.")
            try:
                values.append(float(text))
            except ValueError:
                raise ValueError(f"'{fname}' must be a numeric value.")
        return np.array([values])   # shape (1, 4)

    def evaluate_candidate(self):
        try:
            X_reg = self.get_reg_features()

            reg_model = loaded_models.get("regression")
            if reg_model is None:
                raise RuntimeError("Regression model not loaded.")
            perf_pred = reg_model.predict(X_reg)[0]

            self.reg_result.config(text=f"{perf_pred:.2f}")

            # build classifier features: 4 inputs + predicted performance_score
            X_clf = np.concatenate([X_reg, np.array([[perf_pred]])], axis=1)

            model_name = self.model_var.get()
            clf_model = loaded_models.get(model_name)
            if clf_model is None:
                raise RuntimeError(f"{model_name} model not loaded.")

            clf_pred = clf_model.predict(X_clf)[0]
            label = HIRE_LABELS.get(int(clf_pred), str(clf_pred))

            self.clf_result.config(text=f"{label}  ({model_name})")
            self.status.config(text="Prediction completed successfully.")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.config(text="Error while predicting. See dialog.")

    def clear_inputs(self):
        for entry in self.entries:
            entry.delete(0, tk.END)
        self.reg_result.config(text="—")
        self.clf_result.config(text="—")
        self.status.config(text="Cleared all fields. (press Esc to exit full screen)")


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartHiringApp(root)
    root.mainloop()

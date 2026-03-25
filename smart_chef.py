import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading

# --- PDF LIBRARIES ---
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# --- COLOR PALETTE & FONTS ---
THEME = {
    "bg": "#ffffff",           
    "primary": "#2E7D32",      
    "secondary": "#f5f5f5",    
    "accent": "#FF7043",       
    "text": "#333333",         
    "exit_bg": "#feecec",      
    "exit_fg": "#c62828",      
    "pdf_btn_bg": "#E3F2FD",   # Light Blue for PDF button
    "pdf_btn_fg": "#1565C0",   # Dark Blue text
    "font_header": ("Segoe UI", 24, "bold"),
    "font_sub": ("Segoe UI", 14),
    "font_body": ("Segoe UI", 11)
}

class SmartChefApp:
    def __init__(self, root):
        self.root = root
        self.root.title("👨‍🍳 Smart Chef AI")
        self.root.geometry("900x700")
        self.root.configure(bg=THEME["bg"])

        self.df = None
        self.tfidf_matrix = None
        self.tfidf = None
        self.is_loading = True

        self.create_header()
        self.create_search_area()
        self.create_results_area()
        self.create_footer()

        loader_thread = threading.Thread(target=self.load_data, daemon=True)
        loader_thread.start()
        
        self.check_loading_status()

    def create_header(self):
        header_frame = tk.Frame(self.root, bg=THEME["bg"], pady=20)
        header_frame.pack(fill="x")
        
        title = tk.Label(header_frame, text="👨‍🍳 Smart Chef AI", 
                         font=THEME["font_header"], bg=THEME["bg"], fg=THEME["primary"])
        title.pack()
        
        subtitle = tk.Label(header_frame, text="Tell me what's in your fridge, and I'll tell you what to cook.", 
                            font=THEME["font_body"], bg=THEME["bg"], fg="#666666")
        subtitle.pack()

    def create_search_area(self):
        search_frame = tk.Frame(self.root, bg=THEME["secondary"], pady=20, padx=20)
        search_frame.pack(fill="x", padx=40, pady=10)
        
        tk.Label(search_frame, text="Enter Ingredients (comma separated):", 
                 font=("Segoe UI", 10, "bold"), bg=THEME["secondary"], fg=THEME["text"]).pack(anchor="w")

        self.entry = tk.Entry(search_frame, font=("Segoe UI", 14), bd=2, relief="flat", highlightthickness=1, highlightbackground="#cccccc")
        self.entry.pack(fill="x", pady=(5, 15), ipady=5)
        self.entry.bind('<Return>', self.get_recommendations)
        self.entry.focus_set()

        btn_frame = tk.Frame(search_frame, bg=THEME["secondary"])
        btn_frame.pack(fill="x")

        self.btn_search = tk.Button(btn_frame, text="Find Recipes 🔍", command=self.get_recommendations,
                                    font=("Segoe UI", 11, "bold"), bg=THEME["primary"], fg="white",
                                    relief="flat", cursor="hand2", state="disabled", padx=20, pady=5)
        self.btn_search.pack(side="left")

        self.btn_clear = tk.Button(btn_frame, text="Clear", command=self.clear_input,
                                   font=("Segoe UI", 11), bg="#e0e0e0", fg="black",
                                   relief="flat", cursor="hand2", padx=15, pady=5)
        self.btn_clear.pack(side="left", padx=10)

        self.btn_exit = tk.Button(btn_frame, text="Exit", command=self.close_app,
                                  font=("Segoe UI", 11), bg=THEME["exit_bg"], fg=THEME["exit_fg"],
                                  relief="flat", cursor="hand2", padx=15, pady=5)
        self.btn_exit.pack(side="left", padx=10)

        self.lbl_status = tk.Label(btn_frame, text="Initializing Brain...", font=("Segoe UI", 10, "italic"), 
                                   bg=THEME["secondary"], fg=THEME["accent"])
        self.lbl_status.pack(side="right")

    def create_results_area(self):
        results_frame = tk.LabelFrame(self.root, text=" Recommended Dishes ", 
                                      font=("Segoe UI", 12, "bold"), bg=THEME["bg"], fg=THEME["text"], bd=0)
        results_frame.pack(fill="both", expand=True, padx=40, pady=10)

        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(results_frame, font=("Segoe UI", 12), bg="#fafafa", fg="#333",
                                  bd=0, selectbackground=THEME["primary"], selectforeground="white",
                                  activestyle="none", height=10, yscrollcommand=scrollbar.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        self.listbox.bind('<<ListboxSelect>>', self.show_details)

    def create_footer(self):
        footer = tk.Label(self.root, text="Pro Tip: Click on a recipe to see instructions!", 
                          font=("Segoe UI", 9), bg=THEME["bg"], fg="#999999", pady=10)
        footer.pack(side="bottom")

    def check_loading_status(self):
        if self.is_loading:
            current_text = self.lbl_status.cget("text")
            if "..." in current_text:
                self.lbl_status.config(text="Initializing Brain")
            else:
                self.lbl_status.config(text=current_text + ".")
            self.root.after(500, self.check_loading_status)
        else:
            self.lbl_status.config(text="Ready to cook!", fg=THEME["primary"])
            self.btn_search.config(state="normal")

    def load_data(self):
        try:
            self.df = pd.read_csv('recipes.csv')
            
            rename_map = {}
            if 'Title' in self.df.columns: rename_map['Title'] = 'name'
            if 'Cleaned_Ingredients' in self.df.columns: rename_map['Cleaned_Ingredients'] = 'ingredients'
            if 'Instructions' in self.df.columns: rename_map['Instructions'] = 'instructions'
            
            self.df = self.df.rename(columns=rename_map)
            
            if 'name' not in self.df.columns: self.df['name'] = "Unknown Recipe"
            if 'ingredients' not in self.df.columns: self.df['ingredients'] = ""
            if 'instructions' not in self.df.columns: self.df['instructions'] = "No instructions available."

            self.df['ingredients'] = self.df['ingredients'].astype(str).fillna('')
            self.df['instructions'] = self.df['instructions'].astype(str).fillna('')

            self.tfidf = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.tfidf.fit_transform(self.df['ingredients'])

            self.is_loading = False

        except FileNotFoundError:
            self.is_loading = False
            messagebox.showerror("Critical Error", "File 'recipes.csv' not found!\nPlease place it in the folder.")
        except Exception as e:
            self.is_loading = False
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

    def get_recommendations(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input:
            messagebox.showwarning("Empty Pantry", "Please enter at least one ingredient!")
            return

        user_vec = self.tfidf.transform([user_input])
        scores = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
        
        top_indices = scores.argsort()[-20:][::-1]

        self.listbox.delete(0, tk.END)
        self.current_results = []

        count = 0
        for idx in top_indices:
            if scores[idx] > 0.1:
                row = self.df.iloc[idx]
                self.current_results.append(row)
                self.listbox.insert(tk.END, f" {row['name']}") 
                count += 1
        
        if count == 0:
            self.listbox.insert(tk.END, " No recipes found. Try simplified ingredients.")

    def clear_input(self):
        self.entry.delete(0, tk.END)
        self.listbox.delete(0, tk.END)
        self.entry.focus_set()

    def close_app(self):
        if messagebox.askokcancel("Quit", "Do you really want to exit Smart Chef?"):
            self.root.destroy()

    # --- PDF GENERATION FUNCTION ---
    def generate_pdf(self, recipe):
        try:
            # 1. Ask user where to save the file
            file_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                       filetypes=[("PDF files", "*.pdf")],
                                                       initialfile=f"{recipe['name'][:20]}_Recipe.pdf")
            if not file_path:
                return # User cancelled

            # 2. Setup PDF Document
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # 3. Define Custom Styles
            title_style = styles['Title']
            title_style.textColor = colors.HexColor(THEME["primary"])
            
            header_style = ParagraphStyle('Header', parent=styles['Heading2'], textColor=colors.HexColor(THEME["accent"]), spaceAfter=10)
            body_style = styles['BodyText']
            body_style.leading = 14 # Line spacing

            # 4. Build Content
            # Title
            story.append(Paragraph(recipe['name'], title_style))
            story.append(Spacer(1, 20))

            # Ingredients Section
            story.append(Paragraph("Ingredients", header_style))
            # Clean up the list format if it looks like python list ['item', 'item']
            clean_ing = recipe['ingredients'].replace("[", "").replace("]", "").replace("'", "")
            story.append(Paragraph(clean_ing, body_style))
            story.append(Spacer(1, 20))

            # Instructions Section
            story.append(Paragraph("Instructions", header_style))
            story.append(Paragraph(recipe['instructions'], body_style))

            # 5. Save
            doc.build(story)
            messagebox.showinfo("Success", f"PDF saved successfully at:\n{file_path}")

        except Exception as e:
            messagebox.showerror("PDF Error", f"Could not save PDF:\n{str(e)}")

    def show_details(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return

        index = selection[0]
        if index >= len(self.current_results): return
        
        recipe = self.current_results[index]

        # --- POPUP WINDOW ---
        top = tk.Toplevel(self.root)
        top.title(recipe['name'])
        top.geometry("700x650")
        top.configure(bg="white")

        # Header
        tk.Label(top, text=recipe['name'], font=("Segoe UI", 18, "bold"), 
                 bg="white", fg=THEME["primary"], wraplength=650).pack(pady=15)

        # Button Frame
        btn_frame = tk.Frame(top, bg="white")
        btn_frame.pack(pady=5)

        # Copy Button
        def copy_to_clipboard():
            content = f"Recipe: {recipe['name']}\n\nIngredients:\n{recipe['ingredients']}\n\nInstructions:\n{recipe['instructions']}"
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            messagebox.showinfo("Copied", "Recipe details copied to clipboard!")

        tk.Button(btn_frame, text="📋 Copy Text", command=copy_to_clipboard,
                  bg=THEME["secondary"], relief="flat", font=("Segoe UI", 10)).pack(side="left", padx=5)

        # --- PDF Button ---
        tk.Button(btn_frame, text="📄 Save as PDF", command=lambda: self.generate_pdf(recipe),
                  bg=THEME["pdf_btn_bg"], fg=THEME["pdf_btn_fg"], relief="flat", font=("Segoe UI", 10, "bold")).pack(side="left", padx=5)

        # Content Frame
        content_frame = tk.Frame(top, bg="white", padx=20, pady=10)
        content_frame.pack(fill="both", expand=True)

        # Ingredients
        tk.Label(content_frame, text="Ingredients", font=("Segoe UI", 12, "bold"), 
                 bg="white", fg=THEME["accent"]).pack(anchor="w")
        
        ing_text = tk.Text(content_frame, height=4, font=("Segoe UI", 10), 
                           bg="#f9f9f9", bd=0, padx=10, pady=10)
        ing_text.insert(tk.END, recipe['ingredients'])
        ing_text.config(state="disabled")
        ing_text.pack(fill="x", pady=(0, 15))

        # Instructions
        tk.Label(content_frame, text="Instructions", font=("Segoe UI", 12, "bold"), 
                 bg="white", fg=THEME["accent"]).pack(anchor="w")
        
        inst_text = scrolledtext.ScrolledText(content_frame, font=("Segoe UI", 10), 
                                              bg="#f9f9f9", bd=0, padx=10, pady=10)
        inst_text.insert(tk.END, recipe['instructions'])
        inst_text.config(state="disabled")
        inst_text.pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
        
    app = SmartChefApp(root)
    root.mainloop()
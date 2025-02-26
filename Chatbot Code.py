#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install openai')


# In[6]:


import openai
import tkinter as tk
from tkinter import messagebox, scrolledtext


# In[7]:


OPENAI_API_KEY = "sk-proj-77uely_AGdx-JUDO2TjNeeB1xrp1sZ9lrsvzb2dT3kshBlvhnWMcvkPZJ8Ul"


# In[9]:


openai.api_key = OPENAI_API_KEY


# In[10]:


def get_diet_recommendation(category):
    """ Fetch AI-powered diet recommendations dynamically. """
    prompt = f"Provide a detailed diet plan for someone with {category}. Include meal suggestions and nutritional tips."
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error fetching recommendation: {e}"


# In[ ]:


import tkinter as tk
from tkinter import messagebox, scrolledtext

def submit_choice():
    condition = var.get()
    if condition:
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END) 
        result_text.insert(tk.END, f"AI Diet Suggestions for {condition}:\n\n[Sample AI suggestions here]")
        result_text.config(state=tk.DISABLED)
    else:
        messagebox.showwarning("Input Error", "Please select a condition to get diet suggestions.")

def setup_ui():
    """ Set up interactive UI for Mama Pulse chatbot """
    root = tk.Tk()
    root.title("Mama Pulse - AI Diet Suggestion")
    root.geometry("600x500")
    root.configure(bg="#fce4ec")

    # Title label
    title_label = tk.Label(root, text="Welcome to Mama Pulse", font=("Arial", 20, "bold"), bg="#fce4ec", fg="#d81b60")
    title_label.pack(pady=20)

    # Instruction label
    instruction_label = tk.Label(root, text="Select a condition to get a personalized diet plan:", font=("Arial", 12), bg="#fce4ec")
    instruction_label.pack(pady=10)

    # Condition selection frame
    condition_frame = tk.Frame(root, bg="#fce4ec")
    condition_frame.pack(pady=10)

    conditions = [
        "Pregnancy", "PCOS", "Undergoing IVF", 
        "Gestational Diabetes", "Menopause", "Thyroid Issues"
    ]
    
    global var
    var = tk.StringVar()

    # Listbox for condition selection
    listbox = tk.Listbox(condition_frame, height=6, selectmode=tk.SINGLE, font=("Arial", 12), width=30)
    for condition in conditions:
        listbox.insert(tk.END, condition)
    listbox.pack()

    def on_condition_select(event):
        selection = listbox.curselection()
        if selection:
            var.set(listbox.get(selection[0]))

    listbox.bind('<<ListboxSelect>>', on_condition_select)

    # Submit button
    submit_button = tk.Button(root, text="Get AI-Powered Diet Suggestion", command=submit_choice, 
                              bg="#f48fb1", fg="white", font=("Arial", 14, "bold"), relief="raised")
    submit_button.pack(pady=15)

    # Result section
    result_label = tk.Label(root, text="AI Diet Suggestions:", font=("Arial", 14, "bold"), bg="#fce4ec", fg="#c2185b")
    result_label.pack(pady=5)

    global result_text
    result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10, state=tk.DISABLED, 
                                            font=("Arial", 12), bg="#f8bbd0", fg="black", relief="sunken")
    result_text.pack(padx=20, pady=5)

    root.mainloop()

if __name__ == "__main__":
    setup_ui()


# In[ ]:





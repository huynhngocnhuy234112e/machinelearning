import tkinter as tk
from tkinter import ttk


class DatasetViewer(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.tree = ttk.Treeview(self, show='headings')
        self.tree.pack(fill=tk.BOTH, expand=True)
        sbx = ttk.Scrollbar(self, orient='horizontal', command=self.tree.xview)
        sby = ttk.Scrollbar(self, orient='vertical', command=self.tree.yview)
        self.tree.configure(xscrollcommand=sbx.set, yscrollcommand=sby.set)
        sbx.pack(fill=tk.X, side=tk.BOTTOM)
        sby.pack(fill=tk.Y, side=tk.RIGHT)

    def load_dataframe(self, df):
        # Clear
        for c in self.tree.get_children():
            self.tree.delete(c)
        self.tree['columns'] = list(df.columns)
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=140, stretch=True)
        for _, row in df.iterrows():
            self.tree.insert('', tk.END, values=list(row.values))
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import streamlit as st
import chromadb

# Labs
import lab1
import lab2
import lab3
import lab4
import lab5

# Homeworks
import HW.HW1 as hw1
import HW.HW2 as hw2
import HW.HW3 as hw3
import HW.HW4 as hw4
import HW.HW5 as hw5
import HW.HW7 as hw7

# Sidebar Navigation
st.sidebar.title("Navigation")

# First level: choose section
section = st.sidebar.radio("Choose section:", ["Labs", "HWs"])

if section == "Labs":
    page = st.sidebar.radio("Choose a lab:", ["Lab 1", "Lab 2", "Lab 3", "Lab 4", "Lab 5"])
    if page == "Lab 1":
        lab1.run()
    elif page == "Lab 2":
        lab2.run()
    elif page == "Lab 3":
        lab3.run()
    elif page == "Lab 4":
        lab4.run()
    elif page == "Lab 5":
        lab5.run()

elif section == "HWs":
    page = st.sidebar.radio("Choose a homework:", ["HW1", "HW2", "HW3", "HW4", "HW5", "HW7"])
    if page == "HW1":
        hw1.run()
    elif page == "HW2":
        hw2.run()
    elif page == "HW3":
        hw3.run()
    elif page == "HW4":
        hw4.run()
    elif page == "HW5":
        hw5.run()
    elif page == "HW7":
        hw7.run()

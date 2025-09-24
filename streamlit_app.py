import streamlit as st
import lab1
import lab2
import lab3
import lab4
import HW.HW1 as hw1
import HW.HW2 as hw2
import HW.HW3 as hw3
import HW.HW4 as hw4

st.sidebar.title("Navigation")

# First level: choose section
section = st.sidebar.radio("Choose section:", ["Labs", "HWs"])

if section == "Labs":
    page = st.sidebar.radio("Choose a lab:", ["Lab 1", "Lab 2", "Lab 3", "Lab 4"])
    if page == "Lab 1":
        lab1.run()
    elif page == "Lab 2":
        lab2.run()
    elif page == "Lab 3":
        lab3.run()
    elif page == "Lab 4":
        lab4.run()

elif section == "HWs":
    page = st.sidebar.radio("Choose a homework:", ["HW1", "HW2", "HW3", "HW4"])
    if page == "HW1":
        hw1.run()
    elif page == "HW2":
        hw2.run()
    elif page == "HW3":
        hw3.run()
    elif page == "HW4":
        hw4.run()


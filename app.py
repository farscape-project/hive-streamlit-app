import hydralit_components as hc
import streamlit as st
from navigation.infer_page import application_page
from navigation.home_page import home_page
from navigation.moose_setup_page import moose_setup_page

HOME = 'Home'
APPLICATION = 'HIVE surrogate'
MOOSE_INFO = 'MOOSE configuration'


option_data = [
    {'icon': "🏠", 'label': HOME},
    {'icon': "🤖", 'label': APPLICATION},
    {'icon': "🫎", 'label': MOOSE_INFO},
]
over_theme = {'txc_inactive': 'black', 'menu_background': '#F5B7B1', 'txc_active': 'white', 'option_active': '#CD5C5C'}
font_fmt = {'font-class': 'h3', 'font-size': '50%'}


chosen_tab = hc.option_bar(
    option_definition=option_data,
    title='',
    key='PrimaryOptionx',
    override_theme=over_theme,
    horizontal_orientation=True)

if chosen_tab == HOME:
    home_page()
elif chosen_tab == APPLICATION:
    application_page()
elif chosen_tab == MOOSE_INFO:
    moose_setup_page()

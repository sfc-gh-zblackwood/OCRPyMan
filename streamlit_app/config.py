"""

Config file for Streamlit App

"""

from member import Member


TITLE = "OCRpyMan"

TEAM_MEMBERS = [
    Member(
        name="David Beda",
        linkedin_url="https://www.linkedin.com/in/david-beda-b6943336/",
    ),
    Member(
        name="Jean-Paul Bella",
        linkedin_url="https://www.linkedin.com/in/jean-paul-bella-84986a129/",
        github_url="https://github.com/Jpec57",
    ),
    Member("Thibault Joassard"),
]

PROMOTION = "Promotion Continu Data Scientist - Juin 2022"

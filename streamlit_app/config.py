"""

Config file for Streamlit App

"""

from member import Member


TITLE = "OCRpyMan"

TEAM_MEMBERS = [
    Member(
        name="John Doe",
        linkedin_url="https://www.linkedin.com/in/charlessuttonprofile/",
        github_url="https://github.com/charlessutton",
    ),
    Member("David Beda"),
    Member("Jean-Paul Bella"),
    Member("Thibault Joassard"),
]

PROMOTION = "Promotion Continue Data Scientist - Juin 2021"

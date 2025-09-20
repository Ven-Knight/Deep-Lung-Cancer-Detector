# ────────────────────────────────────────────────────────────────────────────────────────
# Import setuptools for packaging and distribution
# ────────────────────────────────────────────────────────────────────────────────────────
import setuptools

# ────────────────────────────────────────────────────────────────────────────────────────
# Read the long description from README.md for PyPI or GitHub visibility
# ────────────────────────────────────────────────────────────────────────────────────────
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# ────────────────────────────────────────────────────────────────────────────────────────
# Define metadata for the package
# ────────────────────────────────────────────────────────────────────────────────────────
__version__       = "0.0.0"                                # Initial version; update as needed for releases

REPO_NAME         = "Deep-Lung-Cancer-Detector"            # GitHub repository name
AUTHOR_USER_NAME  = "Ven-Knight"                           # GitHub username
SRC_REPO          = "cnnClassifier"                        # Source directory name
AUTHOR_EMAIL      = "venkatareddy.nalamalapu@gmail.com"    # Author contact email

# ────────────────────────────────────────────────────────────────────────────────────────
# Setup function to configure package metadata and structure
# ────────────────────────────────────────────────────────────────────────────────────────
setuptools.setup(
                     name                     = SRC_REPO,                               # Package name
                     version                  = __version__,                            # Package version
                     author                   = AUTHOR_USER_NAME,                       # Author name
                     author_email             = AUTHOR_EMAIL,                           # Author email
                     description              = "A small python package for CNN app",   # Short description
                     long_description         = long_description,                       # Detailed description from README
                     long_description_content = "text/markdown",                        # Format of long description
                     url                      = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",  # Project URL
                     project_urls             = {                                       # Additional project links
                                                   "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
                                                },
                     package_dir              = {"": "src"},                            # Root directory for packages
                     packages                 = setuptools.find_packages(where="src")   # Automatically discover packages in src/
                )
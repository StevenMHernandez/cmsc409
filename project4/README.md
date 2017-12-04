# Setup for Project 4

Project is built in Python3.

Full PDF report is located in `./FINAL_REPORT.pdf`.

Project Code can be found in `./project4.py`

To run the project:

```
# Install dependencies
pip3 install -r requirements.txt
# Run code
python ./project4.py
```

To generate the final report PDF, I ran `./project4.py` which created a Markdown file (see https://daringfireball.net/linked/2014/01/08/markdown-extension)
which is then ran through "PANDOC" (see https://pandoc.org/) 
which can be installed on a Mac with `brew install pandoc`.

To run the markdown file through PANDOC to generate the final pdf,
I ran the following command:

```
pandoc --latex-engine=xelatex -V geometry=margin=1in -s -o FINAL_REPORT.pdf report4.md
```

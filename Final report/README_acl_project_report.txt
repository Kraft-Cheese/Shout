ACL Project Report Draft
========================

Files:
- acl_project_report_draft.tex
- acl_project_report_references.bib

How to compile:
1. For true ACL formatting, place the official ACL files `acl.sty` and
   `acl_natbib.bst` in the same directory, or paste the `.tex` content into
   the official ACL Overleaf template.
2. Compile with:

   pdflatex acl_project_report_draft
   bibtex acl_project_report_draft
   pdflatex acl_project_report_draft
   pdflatex acl_project_report_draft

3. The source currently uses `\usepackage[preprint]{acl}` when `acl.sty` is
   present. Change this to `\usepackage[review]{acl}` for an anonymous review
   version, or `\usepackage[final]{acl}` for a final camera-ready version.

Notes:
- Exact experiment numbers are left as TODO placeholders because the poster
  gives qualitative findings but not all final metric values.
- Fill in the Common Voice version, train/dev/test split, hardware/browser
  setup, adapter hyperparameters, decoding settings, and final WER/CER/F1
  tables before submission.

## Data set

This data was previously studied in paper [Ror2 Signaling and Its Relevance in Breast Cancer Progression](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5483589). The detailed decription can be found in section 1.2 of my thesis.

The file microarray_data.csv contains the gene expression data of breast cancer patients.

The header looks like this:

GeneSymbol,GSM177885,GSM177887,GSM177894,GSM177895,...,GSM615775

The first column contains identifiers of human genes. The names of other columns are patients' IDs.
There are 969 patients and 12179 genes.


The file labels_for_microarray_data.csv contains labels for two classes of patients presented in microarray_data.csv:

1) People who had metastatic event during the first 0-5 years (correspond to "1"). 393 patients.

2) People who did not have metastatic event during the first five years and who had the last follow up between 5 and 10 years. This class corresponds to "0". 576 patients.

**The data files are not included in the repository due to their size.**

# MLCSG-ACL2020
Source code of paper "Modeling Long Context for Task-Oriented Dialogue State Generation" , ACL 2020

## Dependency
Check the packages needed or simply run the command
```console
❱❱❱ pip install -r requirements.txt
```
If you run into an error related to Cython, try to upgrade it first.
```console
❱❱❱ pip install --upgrade cython
```


## Multi-Domain DST
Training
```console
❱❱❱ python3 myTrain.py -separate_label=1 -LanguageModel=1

```
Testing
```console
❱❱❱ python3 myTest.py -path=${save_path}

```

## References
If you use the source codes here in your work, please cite the corresponding papers. The bibtex are listed below:
```
@inproceedings{quan-xiong-2020-modeling,
    title = "Modeling Long Context for Task-Oriented Dialogue State Generation",
    author = "Quan, Jun  and
      Xiong, Deyi",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.637",
    doi = "10.18653/v1/2020.acl-main.637",
    pages = "7119--7124",
}

```

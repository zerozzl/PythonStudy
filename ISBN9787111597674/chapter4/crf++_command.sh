crf_learn -f 4 -p 8 -c 3 /sdb/traindatas/ISBN9787111597674/chapter4/template /sdb/traindatas/ISBN9787111597674/chapter4/train.txt /sdb/traindatas/ISBN9787111597674/chapter4/model>/sdb/traindatas/ISBN9787111597674/chapter4/train.log
crf_test -m /sdb/traindatas/ISBN9787111597674/chapter4/model /sdb/traindatas/ISBN9787111597674/chapter4/test.txt>/sdb/traindatas/ISBN9787111597674/chapter4/test.rst
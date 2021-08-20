#!/bin/bash
cusp_file=$1
accession=$2
echo "$(grep ^[A-Z] ${cusp_file})" > ${cusp_file}
for i in $(seq 1 5)
do
  perl -pi -e 's/\ \ /\ /g' ${cusp_file}
done
perl -pi -e 's/\ /\t/g;s/\*/X/g' ${cusp_file}
for decoded in $(cut -f2 ${cusp_file} | sort -V | uniq)
do
  codon_count=$(awk -v decoded="$decoded" 'BEGIN{FS="\t"}{if($2==decoded){sum+=$5}}END{print sum}' ${cusp_file} )
  if [ "${codon_count}" -gt 0 ]
  then
    awk -v decoded="${decoded}" -v codon_count="${codon_count}" 'BEGIN{FS="\t"}{if($2==decoded){print $1 FS $2 FS $5/codon_count}}' ${cusp_file}
  else
    awk -v decoded="${decoded}" 'BEGIN{FS="\t"}{if($2==decoded){print $1 FS $2 FS 0}}' ${cusp_file}
  fi
done | perl -pe "s/^/$accession\t/;s/X/\*/" | sort -Vk3

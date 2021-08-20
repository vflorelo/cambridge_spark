#!/bin/bash
pepstats_file=$1
block_list=$(grep -wn ^PEPSTATS ${pepstats_file} | cut -d\: -f1)
file_length=$(cat ${pepstats_file} | wc -l)
pos_table=$(paste <(echo "${block_list}") <(echo -e "${block_list}\n${file_length}" | tail -n+2) | awk 'BEGIN{FS="\t"}{print $1 FS $2-$1}')
pos_count=$(echo "${pos_table}" | wc -l )
for pos in $(seq 1 ${pos_count})
do
 start_line=$(echo "${pos_table}" | tail -n+${pos} | head -n1 | cut -f1)
 num_lines=$(echo "${pos_table}" | tail -n+${pos} | head -n1 | cut -f2)
 pepstats_datablock=$(tail -n+${start_line} ${pepstats_file} | head -n${num_lines})
 accession=$(echo "${pepstats_datablock}" | grep -w ^PEPSTATS | awk '{print $3}')
 mol_weight=$(echo "${pepstats_datablock}" | grep -w ^Molecular | awk '{print $4}')
 prot_size=$(echo "${pepstats_datablock}" | grep -w ^Molecular | awk '{print $7}')
 prot_charge=$(echo "${pepstats_datablock}" | grep -w Charge | awk '{print $8}')
 prot_iep=$(echo "${pepstats_datablock}" | grep -w ^Isoelectric | awk '{print $4}')
 tiny_prop=$(echo "${pepstats_datablock}" | grep -w ^Tiny | awk '{print $4}')
 small_prop=$(echo "${pepstats_datablock}" | grep -w ^Small | awk '{print $4}')
 aliphatic_prop=$(echo "${pepstats_datablock}" | grep -w ^Aliphatic | awk '{print $4}')
 aromatic_prop=$(echo "${pepstats_datablock}" | grep -w ^Aromatic | awk '{print $4}')
 nonpolar_prop=$(echo "${pepstats_datablock}" | grep -w ^Non-polar | awk '{print $4}')
 polar_prop=$(echo "${pepstats_datablock}" | grep -w ^Polar | awk '{print $4}')
 charged_prop=$(echo "${pepstats_datablock}" | grep -w ^Charged | awk '{print $4}')
 basic_prop=$(echo "${pepstats_datablock}" | grep -w ^Basic | awk '{print $4}')
 acid_prop=$(echo "${pepstats_datablock}" | grep -w ^Acidic | awk '{print $4}')
 echo -e "${accession}\t${mol_weight}\t${prot_size}\t${prot_charge}\t${prot_iep}\t${tiny_prop}\t${small_prop}\t${aliphatic_prop}\t${aromatic_prop}\t${nonpolar_prop}\t${polar_prop}\t${charged_prop}\t${basic_prop}\t${acid_prop}"
done

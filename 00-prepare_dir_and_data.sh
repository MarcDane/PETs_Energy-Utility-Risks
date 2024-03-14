rm -rf anonymization_effectiveness data dataset_generation energy data_utility results_output
mkdir anonymization_effectiveness data dataset_generation energy data_utility results_output

cd data
mkdir adult student

# Adult
wget https://archive.ics.uci.edu/static/public/2/adult.zip
unzip adult.zip -d adult
rm adult.zip


# Student
wget https://archive.ics.uci.edu/static/public/320/student+performance.zip
unzip student+performance.zip
unzip student.zip -d student
rm student+performance.zip student.zip .student.zip_old

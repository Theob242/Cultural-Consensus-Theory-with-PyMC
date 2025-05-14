# Cultural-Consensus-Theory-with-PyMC
midterm for Cogs107 - spring quarter
## Cultural Consensus Theory Model Summary

In this project, I used PyMC to build a Cultural Consensus Theory (CCT) model that analyzes a dataset about plant knowledge. The dataset includes answers from 10 people to 20 yes/no questions. The model tries to figure out two things: how good each person is at answering correctly (called competence, `D`), and what the most likely correct answer is to each question (called consensus, `Z`). I gave each person a starting range of how competent they might be (between 0.5 and 1.0), and each answer a 50/50 chance of being true. The model learns from the data by sampling thousands of possibilities, and everything ran smoothly with no errors or warnings.

The results showed that some people were more reliable than others — P6 was the most competent, while P3 was the least. The model also gave strong predictions for which answers were most likely correct. I compared the model’s answers to a simple majority vote for each question and found that they matched on 19 out of 20 questions. The one difference (question 2) happened because the model trusted the more reliable informants over the majority. I also saved graphs showing the results in two images: `competence_plot.png` and `consensus_plot.png`. Overall, this model did a better job than just counting votes because it takes into account how trustworthy each person is.


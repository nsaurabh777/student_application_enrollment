# Thanks for your interest in working with us at Emeritus!

To help us understand how you approach real world data science problems, please work through these two cases.

We mostly write Python on our team, but good work is language agnostic, so feel free to use whichever popular language and libraries you prefer.

*R, Julia, Scala, Go, Malbolge, pretty much anything non-proprietary* (as opposed to things like Matlab or SAS which require specific paid software to run) is good with us.

If time starts to drag on, please feel free to simplify your approach and elaborate in text on how the approach could be expanded upon. We don't want to take too much of your time, and are much more interested in how you approach and work through problems than we are that you have the highest performing model.



### Data:
There is both a sqlite db, and a generic sql dump under *db/* so you can use whichever db you prefer. 
- student_profiles contains information about a particular student (potential or actual) who has registered interested in a particular course, uniquely identified by a profile_id. For simplicity's sake, profiles are 1:1 with course and school, and the course and school the student expressed interested in is in their profile. 
- course_applications contains one record for each course and school a student applied to.
- course_enrollments contains one record for each course and school a student enrolled in.
- student_profiles marked with is_test == true are the unannotated test sample. Their application/enrollment state is excluded to provide a basis for analyzing fit. 

###  1. Imagine that we want to segment our marketing campaigns at a country level by performance. 

Build a model that returns which countries we may want to consider together based purely on their performance.

- Describe your model and what led you to this approach.


### 2. Imagine that we want to predict which of these prospective students will apply.

Build a model to predict whether a prospective student will apply to a program (as in, will have a record in student_applications). 

- Describe your model and what led you to this approach.
- What are its most important features?
- What metrics are appropriate for validation on a problem like this? What are the most important tradeoffs.
- Measure and visualize your models performance using whatever metric you think is most relevant.

Please send back code, models, annotations for the test sample, and a short document (markdown, .doc, google doc, etc) addressing the above questions.

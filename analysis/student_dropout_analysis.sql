-- Does marital status has any correlation with the dropout rate?
select Marital_status,count(row_num) as total_students
from student_dropout_data
where Target = 'Dropout'
group by Marital_status

--International Students who got scholarship
select
		count(*) as total_students
from student_dropout_data
where International=1 and Scholarship_holder=1

--How many Students who fall under debtor category  actually got graduated
select count(*) as total_students
from student_dropout_data
where debtor=1 and Target='Graduate'

--Does Mother's & Father's occupation have any connection with the dropout rate?
select top 10 Mother_s_occupation,
		Father_s_occupation,
		Target,
		count(*) as total_students
from student_dropout_data
where Target='Dropout'
group by Mother_s_occupation,Father_s_occupation,Target
order by count(*) desc

--Which course were most taken by females who were categorised under educational special needs?
select Course,
		count(*) as total_students
from student_dropout_data
where Gender='Female' and Educational_special_needs=1
group by Course 
order by count(*) desc

--Nationalities having highest Inflation
select Nacionality,
		avg(Inflation_rate) as avg_inflation_rate,
		count(*) as total_students
from student_dropout_data 
group by Nacionality
order by avg(Inflation_rate) desc

--Nationalities having highest Unemployment rate
select Nacionality,
		avg(Unemployment_rate) as avg_Unemployment_rate,
		count(*) as total_students
from student_dropout_data 
group by Nacionality
order by avg(Unemployment_rate) desc

--The most number of courses taken by students
select Course,
		count(*) as total_students
from student_dropout_data
group by Course
order by count(*) desc

--Does previous qualification of the students have any correlation with the grades obtained in sem 1 and sem 2?
select Previous_qualification,
		avg(curricular_units_1st_sem_grade) as avg_sem1_grade,
		avg(curricular_units_2nd_sem_grade) as avg_sem2_grade,
		count(*) as total_students
from student_dropout_data
group by Previous_qualification
having count(*)>1
order by avg(curricular_units_1st_sem_grade) desc,
		avg(curricular_units_2nd_sem_grade) desc

--Which course results in higher precentage of students graduated
select 
    Course,
        round(
        count(case when Target = 'Graduate' then 1 end) * 100.0 / count(*),0) AS Graduation_Percentage
from student_dropout_data
group by Course
order by Graduation_Percentage desc;

--No of male Students who received admission grade less than 120
select count(*) as total_students
from student_dropout_data
where Gender='Male' and Admission_grade <120

--No of International Students who opted for tourism course
select
		count(*) as total_students
from student_dropout_data
where International=1 and Course='Tourism'

--Count of Students belonging to different Nationalities
select Nacionality,
		count(*) as total_students
from student_dropout_data
group by Nacionality
order by count(*) desc

--What is the overall dropout rate and how does it vary by course?
select 
        round(
        count(case when Target = 'Dropout' then 1 end) * 100.0 / count(*),0) AS Dropout_Percentage
from student_dropout_data
order by Dropout_Percentage desc;

select 
    Course,
        round(
        count(case when Target = 'Dropout' then 1 end) * 100.0 / count(*),0) AS Dropout_Percentage
from student_dropout_data
group by Course
order by Dropout_Percentage desc;

--Which application mode has the highest dropout rate?
select
		Application_mode,
			round(
			count(case when Target = 'Dropout' then 1 end) * 100.0 / count(*),0) AS Dropout_Percentage
from student_dropout_data
group by Application_mode
order by Dropout_Percentage desc

--What is the average 1st sem and 2nd sem grade for Dropout vs Enrolled vs Graduate students?
select Target,
		count(*) as total_students,
		round(avg(Curricular_units_1st_sem_grade),2) as avg_1st_sem_grade,
		round(avg(Curricular_units_2nd_sem_grade),2) as avg_2nd_sem_grade
from student_dropout_data
group by Target
order by count(*) desc

--Does Scholarship status holder correlates with a lower dropout rate?
select Scholarship_holder,
		count(*) as total_students,
		count(case when Target='Dropout' then 1 end)*100.0/count(*) as dropout_rate
from student_dropout_data
group by Scholarship_holder
order by count(*) desc

--Rank the top 5 Courses by dropout rate
select top 5 Course,
		count(*) as total_students,
		count(case when Target='Dropout' then 1 end)*100.0/count(*) as dropout_rate
from student_dropout_data
group by Course
order by dropout_rate desc

--Segment students by Gender and tuition fees upto date status & compute the cumulative count of debtor students who dropped out in each segment
WITH risk_segment AS (
    SELECT 
        CASE 
            WHEN Gender = 'Male'
                 AND Tuition_fees_up_to_date = 0
                 AND Debtor = 1
                 AND Target = 'Dropout'
            THEN 'M_High_risk_dropout'

            WHEN Gender = 'Female'
                 AND Tuition_fees_up_to_date = 0
                 AND Debtor = 1
                 AND Target = 'Dropout'
            THEN 'F_High_risk_dropout'

            ELSE 'Low_risk_dropout'
        END AS risk_category
    FROM student_dropout_data
)
SELECT 
    risk_category,
    COUNT(*) AS total_students
FROM risk_segment
GROUP BY risk_category;

--Age at enrollment bucket cohort analysis of dropout rate-which age bucket has the highest observed dropout rate & does it interact with scholarship holder status?
with bucket_analysis as (
	select Target,
			Scholarship_holder,
		case
			when Age_at_enrollment between 18 and 23
			then '18-23'
			when Age_at_enrollment between 23 and 28
			then '23-28'
			when Age_at_enrollment between 28 and 33
			then '28-33'
			when Age_at_enrollment between 33 and 38
			then '33-38'
			when Age_at_enrollment between 38 and 43
			then '38-43'
			else '43+'
		end as age_group
	from student_dropout_data
)
select age_group,
		Scholarship_holder,
		count(*) as total_students,
		count(case when Target='Dropout' then 1 end)*100.0/count(*) as dropout_rate
from bucket_analysis
group by age_group,Scholarship_holder
order by count(case when Target='Dropout' then 1 end)*100.0/count(*) desc
		



		









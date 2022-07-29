#  Myinh Du 
#  Admissions Project
#  Referenced Dr. Suning Zhu's code (Trinity University)

#  Call relevant packages from library
library(dplyr)
library(tidyverse)
library(caret)
library(e1071)
library(MASS)
library(glmnet)
library(randomForest)
library(gbm)
library(car)
library(DMwR2) 
library(tree)

#  Load original data into R 
clean_data <- read.csv("TU.csv")

#  After skimming through the data (view(clean_data)), note that there are many blanks.
#  Make blanks read as NA. 
#  Decide what to do with NA for each individual variable: how to handle NAs for a variables depends on the nature of that variable.
clean_data[clean_data == ''] <- NA

#  Column 57 - Academic.Index
sum(is.na(clean_data$Academic.Index))#829 NAs
table(clean_data$Academic.Index)#No questionable levels
#  Impute 829 NAs with the most common level
clean_data$Academic.Index[is.na(clean_data$Academic.Index)] <- 3
summary(clean_data$Academic.Index)
summary(factor(clean_data$Academic.Index))
clean_data$Academic.Index <- factor(clean_data$Academic.Index)
summary(clean_data$Academic.Index)

#######Data Cleaning########
#  Column1 - ID
sum(is.na(clean_data$ID)) # No NAs
#  ID should be removed 
clean_data <-subset(clean_data, select = -ID)

#  Column 2 - train.test
sum(is.na(clean_data$train.test))# No NAs
levels(factor(clean_data$train.test))# No suspicious categories.
#  train.test should be removed 
clean_data <-subset(clean_data, select = -train.test)

#  Column 3 - Entry.Term..Application
sum(is.na(clean_data$Entry.Term..Application.)) # No NAs
levels(factor(clean_data$Entry.Term..Application.)) # No suspicious categories
clean_data$Entry.Term..Application. <- as.factor(clean_data$Entry.Term..Application.)
summary(clean_data$Entry.Term..Application.)

#  Column 4 - Admit.Type
sum(is.na(clean_data$Admit.Type)) # No NA 
levels(factor(clean_data$Admit.Type)) # only has one level
# Since the dataset only has first-years (i.e., only one category), Admit.Type should be removed
clean_data <-subset(clean_data, select = -Admit.Type)

#  Column 5 - Permanent.Postal
sum(is.na(clean_data$Permanent.Postal)) # 162 NAs
#Just use "Permanent.Geomarket" and Column 5 may be removed
clean_data <-subset(clean_data, select = -Permanent.Postal)

# Column 6 - Permanent.Country
sum(is.na(clean_data$Permanent.Country))# 1 NA 
levels(factor(clean_data$Permanent.Country))# No suspicious categories
#  The ID with NA in Permanent.Country is 11148. Since this person is a US citizen in the column "Citizenship.Status", Unites States is assigned to NA
clean_data$Permanent.Country[is.na(clean_data$Permanent.Country)] <- "United States"
clean_data$Permanent.Country[clean_data$Permanent.Country!= "United States"] <- "International"
clean_data$Permanent.Country <- as.factor(clean_data$Permanent.Country)

#  Column 7 - Sex
sum(is.na(clean_data$Sex)) # No NAs
levels(factor(clean_data$Sex)) #  No typos
clean_data$Sex <- as.factor(clean_data$Sex)

#  Column 8 - Ethnicity
sum(is.na(clean_data$Ethnicity)) # 227 NAs
levels(factor(clean_data$Ethnicity)) # No questionable category 
clean_data$Ethnicity[is.na(clean_data$Ethnicity)] <- "Not specified"
clean_data$Ethnicity <- as.factor(clean_data$Ethnicity)

#  Column 9 - Race
sum(is.na(clean_data$Race)) # 555 NAs
#  Impute NAs with "Not specified". The reason is similar to that for Ethnicity
clean_data$Race[is.na(clean_data$Race)] <- "Not specified"
levels(factor(clean_data$Race)) # No questionable category
table(clean_data$Race)
#  Current classification of Race is too detailed, leading to very low frequencies for some categories.
#  Need to consider combining some of the categories becausea category with a small number of cases won't have a significant effect on the response.
clean_data$Race <- ifelse(clean_data$Race == "American Indian or Alaska Native", "American Indian or Alaska Native",
                          ifelse(clean_data$Race == "American Indian or Alaska Native, White", "American Indian or Alaska Native, White",
                                 ifelse(clean_data$Race == "Asian", "Asian",
                                        ifelse(clean_data$Race == "Asian, White", "Asian, White",
                                               ifelse(clean_data$Race == "Black or African American", "Black or African American",
                                                      ifelse(clean_data$Race == "Black or African American, White", "Black or African American, White",
                                                             ifelse(clean_data$Race == "Not specified", "Not specified",
                                                                    ifelse(clean_data$Race == "White", "White", "Other"))))))))
clean_data$Race <- as.factor(clean_data$Race)


#  Column 10 - Religion
sum(is.na(clean_data$Religion)) # 5483 NAs
levels(factor(clean_data$Religion))  # No questionable categories
#  Because no religion and other are already included in current levels, it is more reasonable to impute NAs with "Not specified".
clean_data$Religion[is.na(clean_data$Religion)] <- "Not specified"
table(clean_data$Religion)
#  Religion has lots of options, with some options having a very small number of cases. 
#  Combine levels with less than 100 cases into "Other" because a level accounting for lower than 1% of training set is very unlikely to have a significant effect on the response.
clean_data$Religion <- ifelse(clean_data$Religion == "Anglican", "Anglican",
                              ifelse(clean_data$Religion == "Baptist", "Baptist",
                                     ifelse(clean_data$Religion == "Bible Churches", "Christian",
                                            ifelse(clean_data$Religion == "Buddhism", "Buddhism",
                                                   ifelse(clean_data$Religion == "Christian", "Christian",
                                                          ifelse(clean_data$Religion == "Christian Reformed", "Christian",
                                                                 ifelse(clean_data$Religion == "Christian Scientist", "Christian",
                                                                        ifelse(clean_data$Religion == "Church of Christ", "Christian",
                                                                               ifelse(clean_data$Religion == "Church of God", "Christian",
                                                                                      ifelse(clean_data$Religion == "Hindu", "Hindu",
                                                                                             ifelse(clean_data$Religion == "Islam/Muslim", "Islam/Muslim",
                                                                                                    ifelse(clean_data$Religion == "Jewish", "Jewish",
                                                                                                           ifelse(clean_data$Religion == "Lutheran", "Lutheran",
                                                                                                                  ifelse(clean_data$Religion == "Methodist", "Methodist",
                                                                                                                         ifelse(clean_data$Religion == "Not specified", "Not specified",
                                                                                                                                ifelse(clean_data$Religion == "Non-Denominational", "Non-Denominational",
                                                                                                                                       ifelse(clean_data$Religion == "None", "None", 
                                                                                                                                              ifelse(clean_data$Religion == "Presbyterian", "Presbyterian",
                                                                                                                                                     ifelse(clean_data$Religion == "Presbyterian Church of America", "Presbyterian",
                                                                                                                                                            ifelse(clean_data$Religion == "Roman Catholic", "Roman Catholic", "Other"))))))))))))))))))))
clean_data$Religion <- as.factor(clean_data$Religion)

#  Column 11 - First_Source.Origin.First.Source.Date
sum(is.na(clean_data$First_Source.Origin.First.Source.Date)) # No NAs
clean_data$First_Source.Origin.First.Source.Date <- as.Date(clean_data$First_Source.Origin.First.Source.Date, 
                                                            format="%m/%d/%Y")

#  Column 12 - Inquiry.Date
sum(is.na(clean_data$Inquiry.Date)) # 4579 NAs
#  Deal with NAs later. Need to use this variable to create several new variables.
clean_data$Inquiry.Date <- as.Date(clean_data$Inquiry.Date, format="%m/%d/%Y")

#  Column 13 - Submitted
sum(is.na(clean_data$Submitted)) # No NAs
clean_data$Submitted <- as.Date(clean_data$Submitted, format="%m/%d/%Y")

#  Column 11-13
#  After viewing Column 11-13, it would be interesting to see whether the differences between submission date and First_Source date and the differences between submission date and inquiry date affect the response.
clean_data$Submit_FirstSource <- difftime(clean_data$Submitted, 
                                          clean_data$First_Source.Origin.First.Source.Date, 
                                          units = "weeks")
clean_data$Submit_Inquiry <- difftime(clean_data$Submitted, 
                                      clean_data$Inquiry.Date, units = "weeks")
clean_data$Submit_FirstSource <- round(clean_data$Submit_FirstSource, digits = 0)
clean_data$Submit_FirstSource <- as.numeric(clean_data$Submit_FirstSource)
clean_data$Submit_Inquiry <- round(clean_data$Submit_Inquiry, digits = 0)
clean_data$Submit_Inquiry <- as.numeric(clean_data$Submit_Inquiry)
#  There are NAs in Inquiry.Date, leading to NAs in Submit_Inquiry.
#  Impute NAs in Submit_Inquiry with median values.
clean_data$Submit_Inquiry[is.na(clean_data$Submit_Inquiry)] <- median(clean_data$Submit_Inquiry,
                                                                      na.rm=TRUE)
#  Remove Column 11-13 (since they are used to construct new variables).  
clean_data <-subset(clean_data, select = -First_Source.Origin.First.Source.Date)
clean_data <-subset(clean_data, select = -Inquiry.Date)
clean_data <-subset(clean_data, select = -Submitted)

#  Column  14 - Application.Source
sum(is.na(clean_data$Application.Source)) # No NAs
table(clean_data$Application.Source) # No questionable categories
clean_data$Application.Source <- as.factor(clean_data$Application.Source)

#  Column  15 - Decision.Plan
sum(is.na(clean_data$Decision.Plan)) # No NAs
table(clean_data$Decision.Plan) #No questionable categories
clean_data$Decision.Plan <- as.factor(clean_data$Decision.Plan)    

#  Column  16 - Staff.Assigned.Name
#  Remove, as staff is likely unhelpful in modeling. In additon, staff has changed over time so this variable has mixed categories depending on the year. 
clean_data <-subset(clean_data, select = -Staff.Assigned.Name)

#  Column  17 - Legacy
sum(is.na(clean_data$Legacy)) # 13658 NAs.
table(clean_data$Legacy) # No questionable categories
#  Impute NAs with "No Legacy"
clean_data$Legacy[is.na(clean_data$Legacy)] <- "No Legacy"
#  Legacy has many options, leading some options to having only a small number of cases. Group all the options into 3 categories so that each category has the chance to affect the response.
clean_data$Legacy <- ifelse(clean_data$Legacy == "Legacy", "Legacy", 
                            ifelse(clean_data$Legacy == "No Legacy", "No Legacy",
                                   ifelse(grepl("Legacy, Opt Out",clean_data$Legacy), 
                                          "Legacy, Opt Out", "Legacy")))
clean_data$Legacy <- as.factor(clean_data$Legacy)

#  Column 18 - Athlete
sum(is.na(clean_data$Athlete)) # 13120 NAs.
table(clean_data$Athlete) # No questionable category.
#  Impute NAs with "Non-Athlete"
clean_data$Athlete[is.na(clean_data$Athlete)] <- "Non-Athlete"
#  Similar to Column 17, Column 18 has many categories with a few cases.
#  Group all options into three categories: Athlete, Non-Athlete, and Athlete, Opt Out.
clean_data$Athlete <- ifelse(clean_data$Athlete == "Athlete", "Athlete", 
                             ifelse(clean_data$Athlete == "Non-Athlete", "Non-Athlete",
                                    ifelse(grepl("Opt Out",clean_data$Athlete), 
                                           "Athlete, Opt Out", "Athlete")))
clean_data$Athlete <- as.factor(clean_data$Athlete)                                                                                                                                                                                               

#  Column 19 - Sport.1.Sport
sum(is.na(clean_data$Sport.1.Sport)) # 13120 NAs.
table(clean_data$Sport.1.Sport) # No questionable categories
#  Impute NAs with "No Sport".
clean_data$Sport.1.Sport[is.na(clean_data$Sport.1.Sport)] <- "No Sport"
#  Group sport men and sport women into one group so that each group has sufficient cases to have an impact on the response.
clean_data$Sport.1.Sport <- ifelse(clean_data$Sport.1.Sport == "Baseball", "Baseball", 
                                   ifelse(clean_data$Sport.1.Sport == "Softball", "Softball",
                                          ifelse(clean_data$Sport.1.Sport == "Football", "Football", 
                                                 ifelse(clean_data$Sport.1.Sport == "No Sport", "No Sport", 
                                                        ifelse(grepl("Basketball", clean_data$Sport.1.Sport), "Basketball",
                                                               ifelse(grepl("Cross Country", clean_data$Sport.1.Sport), "Cross Country",
                                                                      ifelse(grepl("Diving", clean_data$Sport.1.Sport), "Diving",
                                                                             ifelse(grepl("Golf", clean_data$Sport.1.Sport), "Golf",
                                                                                    ifelse(grepl("Soccer", clean_data$Sport.1.Sport), "Soccer",
                                                                                           ifelse(grepl("Swimming", clean_data$Sport.1.Sport), "Swimming",
                                                                                                  ifelse(grepl("Tennis", clean_data$Sport.1.Sport), "Tennis",
                                                                                                         ifelse(grepl("Track", clean_data$Sport.1.Sport), "Track", "Volleyball"))))))))))))
clean_data$Sport.1.Sport <- as.factor(clean_data$Sport.1.Sport)
summary(clean_data$Sport.1.Sport)

#  Column 20 - Sport.1.Rating
sum(is.na(clean_data$Sport.1.Rating)) # 13120 NAs.
table(clean_data$Sport.1.Rating) # No questionable categories.
#  Impute NAs with "No Sport".
clean_data$Sport.1.Rating[is.na(clean_data$Sport.1.Rating)] <- "No Sport"
clean_data$Sport.1.Rating<- factor(clean_data$Sport.1.Rating, order = TRUE, 
                                   levels = c("No Sport", "Varsity", "Blue Chip", "Franchise"))
summary(clean_data$Sport.1.Rating)

#  Column 21 - Sport.2.Sport
sum(is.na(clean_data$Sport.2.Sport)) # 14513 NAs.
table(clean_data$Sport.2.Sport) # No questionable categories
#  impute NAs with "No 2ndSport".
clean_data$Sport.2.Sport[is.na(clean_data$Sport.2.Sport)] <- "No 2ndSport"
#The number of cases for each sport type is very small (< about 1% of the data set).
#It's better to group all options into 2 categories: 2ndSport vs. No 2ndSport.
clean_data$Sport.2.Sport <- ifelse(clean_data$Sport.2.Sport == "No 2ndSport", 
                                   "No 2ndSport", "2ndSport")
clean_data$Sport.2.Sport <- as.factor(clean_data$Sport.2.Sport)

#  Column 22 - Sport.2.Rating
sum(is.na(clean_data$Sport.2.Rating)) # 15085 NAs.
table(clean_data$Sport.2.Rating) # No questionable categories
#  Only 58 out of 15143 observations are rated. This is less than 0.5% of the data set! I don't think Sport.2.Rating will have much impact on the response.
#  Remove
clean_data <-subset(clean_data, select = -Sport.2.Rating)

#  Column 23 - Sport.3.Sport
sum(is.na(clean_data$Sport.3.Sport)) # 14907 NAs.
table(clean_data$Sport.3.Sport) # No questionable categories
#  impute NAs with "No 3rdSport".
clean_data$Sport.3.Sport[is.na(clean_data$Sport.3.Sport)] <- "No 3rdSport"
#  The number of cases for each sport type is very small (< 0.5% of the data set).
#  It's better to group all options into 2 categories: 3rdSport vs. No 3rdSport.
clean_data$Sport.3.Sport <- ifelse(clean_data$Sport.3.Sport == "No 3rdSport", 
                                   "No 3rdSport", "3rdSport")
clean_data$Sport.3.Sport <- as.factor(clean_data$Sport.3.Sport)

#  Column 24 - Sport.3.Rating
sum(is.na(clean_data$Sport.3.Rating)) # 15140 NAs.
table(clean_data$Sport.3.Rating) # No questionable category.
#  Only 3 out of 15143 observations are rated
#  Consider removing Column 24 in the modeling stage.
clean_data <-subset(clean_data, select = -Sport.3.Rating)


#  Column 25 - Academic.Interest.1
sum(is.na(clean_data$Academic.Interest.1)) # 6 NAs.
table(clean_data$Academic.Interest.1) # No questionable category.
clean_data[is.na(clean_data$Academic.Interest.1),]
#  Most of the NAs for Academic.Interest.1 have a value for Academic.Interest.2
#  Assign the corresponding values in Academic.Interest.2 to NAs in Academic.Interest.1 if Academic.Interest.2 has a value.
clean_data$Academic.Interest.1 <- ifelse(is.na(clean_data$Academic.Interest.1) == TRUE, 
                                         clean_data$Academic.Interest.2, 
                                         clean_data$Academic.Interest.1)
#  For the remaining NAs in Academic.Interest.1, Undecided.
clean_data$Academic.Interest.1[is.na(clean_data$Academic.Interest.1)] <- "Undecided"
clean_data$Academic.Interest.1 <- ifelse(grepl("Business", clean_data$Academic.Interest.1), "Business",
                                         ifelse(clean_data$Academic.Interest.1 == "Finance", "Business",
                                                ifelse(clean_data$Academic.Interest.1 == "Entrepreneurship", "Business", 
                                                       clean_data$Academic.Interest.1)))
#  Group options with a low number of cases (< 100 cases) into "Others".
frequencies <-data.frame(table(clean_data$Academic.Interest.1))
frequencies
clean_data$Academic.Interest.1.Frequency <- NA
for (i in 1:nrow(clean_data)){
  for(j in 1:nrow(frequencies)){
    if (clean_data$Academic.Interest.1[i] == frequencies$Var1[j])
    {clean_data$Academic.Interest.1.Frequency[i] <- frequencies$Freq[j]}}
}

for (i in 1:nrow(clean_data)){
  if (clean_data$Academic.Interest.1.Frequency[i] < 100)
  {clean_data$Academic.Interest.1[i] <- "Other"}else{
    clean_data$Academic.Interest.1[i]
  }
}
clean_data$Academic.Interest.1 <- as.factor(clean_data$Academic.Interest.1)
#Drop Academic.Interest.1.Frequency 
clean_data <-subset(clean_data, select = -Academic.Interest.1.Frequency)

#  Column 26 - Academic.Interest.2
sum(is.na(clean_data$Academic.Interest.2)) # 159 NAs.
# Replace repeated academic interests with Undecided, then make NAs Undecided
clean_data$Academic.Interest.2 <- ifelse(clean_data$Academic.Interest.2 == clean_data$Academic.Interest.1, 
                                         "Undecided", clean_data$Academic.Interest.2)
clean_data$Academic.Interest.2[is.na(clean_data$Academic.Interest.2)] <- "Undecided"
table(clean_data$Academic.Interest.2) # No questionable categories.
#  Group Business related options into "Business".
clean_data$Academic.Interest.2 <- ifelse(grepl("Business", clean_data$Academic.Interest.2), "Business",
                                         ifelse(clean_data$Academic.Interest.2 == "Finance", "Business",
                                                ifelse(clean_data$Academic.Interest.2 == "Entrepreneurship", "Business", 
                                                       clean_data$Academic.Interest.2)))
#  Group options with a low number of cases (<100 cases) into "Others".
frequencies <-data.frame(table(clean_data$Academic.Interest.2))
frequencies
clean_data$Academic.Interest.2.Frequency <- NA
for (i in 1:nrow(clean_data)){
  for(j in 1:nrow(frequencies)){
    if (clean_data$Academic.Interest.2[i] == frequencies$Var1[j])
    {clean_data$Academic.Interest.2.Frequency[i] <- frequencies$Freq[j]}}
}

for (i in 1:nrow(clean_data)){
  if (clean_data$Academic.Interest.2.Frequency[i] < 100)
  {clean_data$Academic.Interest.2[i] <- "Other"}else{
    clean_data$Academic.Interest.2[i]
  }
}
clean_data$Academic.Interest.2 <- as.factor(clean_data$Academic.Interest.2)
#  Drop Academic.Interest.2.Frequency
clean_data <-subset(clean_data, select = -Academic.Interest.2.Frequency)


#  Column 27 - First_Source.Origin.First.Source.Summary
sum(is.na(clean_data$First_Source.Origin.First.Source.Summary))# No NA.
table(clean_data$First_Source.Origin.First.Source.Summary)# No questionable category.
#  Group options with a low number of cases (< 100) into "Other Sources".
frequencies <-data.frame(table(clean_data$First_Source.Origin.First.Source.Summary))
frequencies
clean_data$First_Source.Summary.Frequency <- NA
for (i in 1:nrow(clean_data)){
  for(j in 1:nrow(frequencies)){
    if (clean_data$First_Source.Origin.First.Source.Summary[i] == frequencies$Var1[j])
    {clean_data$First_Source.Summary.Frequency[i] <- frequencies$Freq[j]}}
}

for (i in 1:nrow(clean_data)){
  if (clean_data$First_Source.Summary.Frequency[i] < 100)
  {clean_data$First_Source.Origin.First.Source.Summary[i] <- "Other Sources"}else{
    clean_data$First_Source.Origin.First.Source.Summary[i]
  }
}
clean_data$First_Source.Origin.First.Source.Summary <- as.factor(clean_data$First_Source.Origin.First.Source.Summary)
#  Drop First_Source.Summary.Frequency 
clean_data <-subset(clean_data, select = -First_Source.Summary.Frequency)


#  Column 28 - Total.Event.Participation
sum(is.na(clean_data$Total.Event.Participation)) # No NAs
table(clean_data$Total.Event.Participation)#No questionable category.
#  3, 4, 5 combined accounts for < 1% of the data set.
#  Compared to the number of cases in 0, 1, and 2, the number of cases
#  in 3, 4, and 5 won't be very useful in predicting the response.
#  Factor the variable and group 3, 4, and 5 into "2 or more".
clean_data$Total.Event.Participation <- ifelse(clean_data$Total.Event.Participation > 2,
                                               2, clean_data$Total.Event.Participation)
#  Convert int to char so that level name can be modified.
clean_data$Total.Event.Participation <- as.character(clean_data$Total.Event.Participation)
clean_data$Total.Event.Participation <- ifelse(clean_data$Total.Event.Participation == "2",
                                               "2 or more", clean_data$Total.Event.Participation)
clean_data$Total.Event.Participation <- as.factor(clean_data$Total.Event.Participation)

#  Column 29 - Count.of.Campus.Visits
sum(is.na(clean_data$Count.of.Campus.Visits))  # No NAs
table(clean_data$Count.of.Campus.Visits) # No questionable categories
#  Factor the variable and group 5, 6, and 8 into 4.
clean_data$Count.of.Campus.Visits <- ifelse(clean_data$Count.of.Campus.Visits > 4,
                                            4, clean_data$Count.of.Campus.Visits)
#  Convert int to char so level name can be modified
clean_data$Count.of.Campus.Visits <- as.character(clean_data$Count.of.Campus.Visits)
clean_data$Count.of.Campus.Visits <- ifelse(clean_data$Count.of.Campus.Visits == "4",
                                            "4 or more", clean_data$Count.of.Campus.Visits)
clean_data$Count.of.Campus.Visits <- as.factor(clean_data$Count.of.Campus.Visits)


#  Column 30 - School..1.Organization.Category
sum(is.na(clean_data$School..1.Organization.Category)) # 38 NAs.
table(clean_data$School..1.Organization.Category) # No questionable categories
#  Only 16 cases belong to College but 15089 cases belong to High School. Should remove this variable.
clean_data <-subset(clean_data, select = -School..1.Organization.Category)

#  Column 31 - School.1.Code
sum(is.na(clean_data$School.1.Code)) # 11879 NAs.
table(clean_data$School.1.Code)
#  Will School Code matter much? Plus,there are 11879 missing values
clean_data <-subset(clean_data, select = -School.1.Code)

#  Column 32 - School.1.Class.Rank..Numeric.
sum(is.na(clean_data$School.1.Class.Rank..Numeric.)) # 8136 NAs.
#  Column 33 - School.1.Class.Size..Numeric.
sum(is.na(clean_data$School.1.Class.Size..Numeric.)) # 8136 NAs.
#  Percentage rank can more accurately reflect a student's academic performance than numeric rank. 
#  New Column - School.1.Top.Percent.in.Class
clean_data$School.1.Top.Percent.in.Class <- NA
clean_data$School.1.Top.Percent.in.Class <- 100 *(clean_data$School.1.Class.Rank..Numeric./clean_data$School.1.Class.Size..Numeric.)

sum(is.na(clean_data$School.1.Top.Percent.in.Class))
#  Impute the 8136 NAs based on Academic.Index column. 
#  Need to handle NAs in School.1.Top.Percent.in.Class according Academic.Index


#  No missing values in Academic.Index now.
#  Impute missing values in School.1.Top.Percent.in.Class based on Academic.Index.
clean_index_1 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 1) %>%
  mutate(School.1.Top.Percent.in.Class = replace(School.1.Top.Percent.in.Class, 
                                                 is.na(School.1.Top.Percent.in.Class), mean(School.1.Top.Percent.in.Class, na.rm=TRUE)))

clean_index_2 <- clean_data %>% 
  group_by(Academic.Index) %>% 
  filter(Academic.Index == 2) %>%
  mutate(School.1.Top.Percent.in.Class = replace(School.1.Top.Percent.in.Class, 
                                                 is.na(School.1.Top.Percent.in.Class), mean(School.1.Top.Percent.in.Class, na.rm=TRUE)))

clean_index_3 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 3) %>%
  mutate(School.1.Top.Percent.in.Class = replace(School.1.Top.Percent.in.Class, 
                                                 is.na(School.1.Top.Percent.in.Class), mean(School.1.Top.Percent.in.Class, na.rm=TRUE)))  

clean_index_4 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 4) %>%
  mutate(School.1.Top.Percent.in.Class = replace(School.1.Top.Percent.in.Class, 
                                                 is.na(School.1.Top.Percent.in.Class), mean(School.1.Top.Percent.in.Class, na.rm=TRUE)))    

clean_index_5 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 5) %>%
  mutate(School.1.Top.Percent.in.Class = replace(School.1.Top.Percent.in.Class, 
                                                 is.na(School.1.Top.Percent.in.Class), mean(School.1.Top.Percent.in.Class, na.rm=TRUE)))     

clean_data <- rbind(clean_index_1, clean_index_2, clean_index_3, clean_index_4, clean_index_5)

#  Should class rank numeric and class size numeric be deleted? 
clean_data <-subset(clean_data, select = -School.1.Class.Rank..Numeric.)
clean_data <-subset(clean_data, select = -School.1.Class.Size..Numeric.)

#  Column 34 - School.1.GPA
#  Remove this variable because School.1.GPA.Recalculated is more accurate.
clean_data <-subset(clean_data, select = -School.1.GPA)

#  Column 35 - School.1.GPA.Scale
#  Remove this variable as it is irrelevant.
clean_data <-subset(clean_data, select = -School.1.GPA.Scale)

#  Column 36 - School.1.GPA.Recalculated
sum(is.na(clean_data$School.1.GPA.Recalculated))#0 NAs
skewness(clean_data$School.1.GPA.Recalculated^3)
summary(clean_data$School.1.GPA.Recalculated)
ggplot(clean_data, aes(School.1.GPA.Recalculated^3)) +geom_density()
clean_data$School.1.GPA.Recalculated <-clean_data$School.1.GPA.Recalculated^3
#  Moderately skewed, consider transformation.

#  Column 37 - School.2.Class.Rank..Numeric.
sum(is.na(clean_data$School.2.Class.Rank..Numeric.))# 15143 NAs.
#  All cases are blank. Remove this variable
clean_data <-subset(clean_data, select = -School.2.Class.Rank..Numeric.)

#  Column  38 - School.2.Class.Size..Numeric.
sum(is.na(clean_data$School.2.Class.Size..Numeric.)) # 15143 NAs.
#  All cases are blank. Remove this variable 
clean_data <-subset(clean_data, select = - School.2.Class.Size..Numeric.)

#  Column  39 - School.2.GPA
sum(is.na(clean_data$School.2.GPA)) # 15143 NAs.
#  All cases are blank. Remove this variable
clean_data <-subset(clean_data, select = - School.2.GPA)

#  Column 40 - School.2.GPA.Scale
sum(is.na(clean_data$School.2.GPA.Scale) )# 15143 NAs.
#  All cases are blank. Remove this variable 
clean_data <-subset(clean_data, select = -School.2.GPA.Scale)

#  Column 41 - School.2.GPA.Recalculated
sum(is.na(clean_data$School.2.GPA.Recalculated)) # 15143 NAs.
#  All cases are blank. Remove this variable 
clean_data <-subset(clean_data, select = -School.2.GPA.Recalculated)

#  Column 42 - School.3.Class.Rank..Numeric.
sum(is.na(clean_data$School.3.Class.Rank..Numeric.)) # 15143 NAs.
#  All cases are blank. Remove this variable
clean_data <-subset(clean_data, select = -School.3.Class.Rank..Numeric.)

# Column 43 - School.3.Class.Size..Numeric.
sum(is.na(clean_data$School.3.Class.Size..Numeric.))#15143 NAs.
#  All cases are blank. Remove this variable
clean_data <-subset(clean_data, select = -School.3.Class.Size..Numeric.)

#  Column 44 - School.3.GPA
sum(is.na(clean_data$School.3.GPA))#15143 NAs.
#  All cases are blank. Remove this variable 
clean_data <-subset(clean_data, select = -School.3.GPA)

#  Column 45 - School.3.GPA.Scale
sum(is.na(clean_data$School.3.GPA.Scale))#15143 NAs.
#  All cases are blank. Remove this variable
clean_data <-subset(clean_data, select = -School.3.GPA.Scale)

#  Column 46 - School.3.GPA.Recalculated
sum(is.na(clean_data$School.3.GPA.Recalculated))#15143 NAs.
#All cases are blank. Remove this variable 
clean_data <-subset(clean_data, select = -School.3.GPA.Recalculated)

#  Column 47 - ACT.Composite
sum(is.na(clean_data$ACT.Composite))# 7502 NAs.
clean_data$ACT.Composite[is.na(clean_data$ACT.Composite)] <- "No Submission"
summary(factor(clean_data$ACT.Composite))
unique(clean_data$ACT.Composite)
clean_data$ACT.Composite <- factor(clean_data$ACT.Composite, order = TRUE, levels = c("No Submission", "15","17", "18", "19", "20", "21", "22", "23","24", "25", "26", "27", "28", "29", "30", "31", "32", "33","34", "35", "36"))
summary(clean_data$ACT.Composite)

#  Column 48 - ACT.English
sum(is.na(clean_data$ACT.English))#7883 NAs
NAACT <- clean_data[is.na(clean_data$ACT.English),]
# 381 of entries that do not report individual ACT subject scores DO report overall ACT composite scores. This means it's likely that Those who have ACT composites did not report their individual subject scores AND those who do not have any ACT composite score did not submit ACT scores at all.  
#7883-381 #perfect hole in the observations of NAs in the ACT.English column matches the number of ACT.Composite NAs
summary(clean_data$ACT.English)

# mpute missing values based on Academic.Index.
clean_index_1 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 1) %>%
  mutate(ACT.English = replace(ACT.English, is.na(ACT.English), mean(ACT.English, na.rm=TRUE)))

clean_index_2 <- clean_data %>% 
  group_by(Academic.Index) %>% 
  filter(Academic.Index == 2) %>%
  mutate(ACT.English = replace(ACT.English, is.na(ACT.English), mean(ACT.English, na.rm=TRUE)))

clean_index_3 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 3) %>%
  mutate(ACT.English = replace(ACT.English, is.na(ACT.English), mean(ACT.English, na.rm=TRUE)))  

clean_index_4 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 4) %>%
  mutate(ACT.English = replace(ACT.English, is.na(ACT.English), mean(ACT.English, na.rm=TRUE)))    

clean_index_5 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 5) %>%
  mutate(ACT.English = replace(ACT.English, is.na(ACT.English), mean(ACT.English, na.rm=TRUE)))     

clean_data <- rbind(clean_index_1, clean_index_2, clean_index_3, clean_index_4, clean_index_5)

summary(clean_data$ACT.English)

######################################################################

#Column 49 - ACT.Reading
sum(is.na(clean_data$ACT.Reading)) # 7883 NAs
#Impute missing values based on Academic.Index.
clean_index_1 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 1) %>%
  mutate(ACT.Reading  = replace(ACT.Reading , is.na(ACT.Reading ), mean(ACT.Reading , na.rm=TRUE)))

clean_index_2 <- clean_data %>% 
  group_by(Academic.Index) %>% 
  filter(Academic.Index == 2) %>%
  mutate(ACT.Reading  = replace(ACT.Reading , is.na(ACT.Reading ), mean(ACT.Reading , na.rm=TRUE)))

clean_index_3 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 3) %>%
  mutate(ACT.Reading  = replace(ACT.Reading , is.na(ACT.Reading ), mean(ACT.Reading , na.rm=TRUE)))  

clean_index_4 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 4) %>%
  mutate(ACT.Reading  = replace(ACT.Reading , is.na(ACT.Reading ), mean(ACT.Reading , na.rm=TRUE)))    

clean_index_5 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 5) %>%
  mutate(ACT.Reading  = replace(ACT.Reading , is.na(ACT.Reading ), mean(ACT.Reading , na.rm=TRUE)))     

clean_data <- rbind(clean_index_1, clean_index_2, clean_index_3, clean_index_4, clean_index_5)
summary(clean_data$ACT.Reading)


#Column 50 - ACT.Math 
sum(is.na(clean_data$ACT.Math))#7883 NAs
#Impute missing values based on Academic.Index.
clean_index_1 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 1) %>%
  mutate(ACT.Math  = replace(ACT.Math , is.na(ACT.Math ), mean(ACT.Math , na.rm=TRUE)))

clean_index_2 <- clean_data %>% 
  group_by(Academic.Index) %>% 
  filter(Academic.Index == 2) %>%
  mutate(ACT.Math  = replace(ACT.Math , is.na(ACT.Math ), mean(ACT.Math , na.rm=TRUE)))

clean_index_3 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 3) %>%
  mutate(ACT.Math  = replace(ACT.Math , is.na(ACT.Math ), mean(ACT.Math , na.rm=TRUE)))  

clean_index_4 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 4) %>%
  mutate(ACT.Math  = replace(ACT.Math , is.na(ACT.Math ), mean(ACT.Math , na.rm=TRUE)))    

clean_index_5 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 5) %>%
  mutate(ACT.Math  = replace(ACT.Math , is.na(ACT.Math ), mean(ACT.Math , na.rm=TRUE)))     

clean_data <- rbind(clean_index_1, clean_index_2, clean_index_3, clean_index_4, clean_index_5)
summary(clean_data$ACT.Math)


# Column 51 - ACT.Science.Reasoning
sum(is.na(clean_data$ACT.Science.Reasoning)) # 7883 NAs
# Impute missing values based on Academic.Index.
clean_index_1 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 1) %>%
  mutate(ACT.Science.Reasoning  = replace(ACT.Science.Reasoning , is.na(ACT.Science.Reasoning ), mean(ACT.Science.Reasoning , na.rm=TRUE)))

clean_index_2 <- clean_data %>% 
  group_by(Academic.Index) %>% 
  filter(Academic.Index == 2) %>%
  mutate(ACT.Science.Reasoning  = replace(ACT.Science.Reasoning , is.na(ACT.Science.Reasoning ), mean(ACT.Science.Reasoning , na.rm=TRUE)))

clean_index_3 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 3) %>%
  mutate(ACT.Science.Reasoning  = replace(ACT.Science.Reasoning , is.na(ACT.Science.Reasoning ), mean(ACT.Science.Reasoning , na.rm=TRUE)))  

clean_index_4 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 4) %>%
  mutate(ACT.Science.Reasoning  = replace(ACT.Science.Reasoning , is.na(ACT.Science.Reasoning ), mean(ACT.Science.Reasoning , na.rm=TRUE)))    

clean_index_5 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 5) %>%
  mutate(ACT.Science.Reasoning  = replace(ACT.Science.Reasoning , is.na(ACT.Science.Reasoning ), mean(ACT.Science.Reasoning , na.rm=TRUE)))     

clean_data <- rbind(clean_index_1, clean_index_2, clean_index_3, clean_index_4, clean_index_5)
summary(clean_data$ACT.Science.Reasoning)
which(is.na(clean_data$ACT.Science.Reasoning))

# Column 52 - ACT.Writing
sum(is.na(clean_data$ACT.Writing)) # 14886 NAs
summary(factor(clean_data$ACT.Writing))
#Not required to take the ACT writing test, even when taking the normal ACT. 
clean_data$ACT.Writing[is.na(clean_data$ACT.Writing)] <- "No ACT Writing Submission"
#Be aware that the 2016-2017 and on ACT writing scale os from 1-12, not 1-36. This means that the 1 observation of a score of 18 is not comparable to the other scores bc they are not scored on the same scale. using an ACT score converter from ACT.org, an old score of 18 equates to a new score of 7. Convert that score
clean_data$ACT.Writing[clean_data$ACT.Writing == "18"] <- "7"
#factor the scores 
clean_data$ACT.Writing <- factor(clean_data$ACT.Writing, order =TRUE, levels = c("No ACT Writing Submission", "5", "6", "7", "8", "9", "10", "11", "12"))
# It should be noted that since reporting the writing ACT score is completely optional, its possible that the reason there are no/very few observations for ACT writing scores at or below a 7 is because those students took the writing ACT, did poorly and chose not to report it. For reference, there is a huge jump in percentile between a 7 and 8. A score of 7 puts you in the 66th percentile, while an 8 puts you in the 90th percentile. 6 is the mean. So people with a score of 7 or lower might not choose to report this score if they think it will make their academic stength look bad, and people who score an 8 or above will report it because is strengthens their academic image. 
#Source: ACT.org, percentiles are based on ACT-tested US  high school graduates of 2019, 2020, and 2021 who took the ACT writing test, n = 1,930,800. 

# Column 53 - SAT.I.CR...M (Uncentered SAT Scores... there shouldn't be very many values for this column)
summary(factor(clean_data$SAT.I.CR...M))
clean_data$SAT.I.CR...M[is.na(clean_data$SAT.I.CR...M)] <- "Not Submitted"
clean_data$SAT.I.CR...M <- factor(clean_data$SAT.I.CR...M, order =TRUE, levels = c("Not Submitted", "800", "830", "870", "880", "910", "930", "940", "960", "970", "980", "990", "1000", "1010", "1020", "1030", "1040", "1045", "1050", "1060", "1070", "1080", "1090", "1100", "1110", "1120", "1130", "1140", "1150", "1160", "1170", "1180", "1190", "1200", "1210", "1220","1230", "1240", "1250", "1260", "1270", "1280", "1290", "1300","1310", "1320", "1330", "1340", "1350", "1360", "1370", "1380", "1390", "1400", "1410", "1420", "1430", "1440", "1450", "1460","1470", "1480", "1490", "1500", "1510", "1520", "1530", "1540","1550", "1560", "1570", "1580", "1590", "1600"))
summary(clean_data$SAT.I.CR...M)
14569/15143
15142-14569
1-.96209
# accounts for 3.71% of values
clean_data <-subset(clean_data, select = -SAT.I.CR...M)


# Column 54 - SAT.R.Evidence.Based.Reading.and.Writing.Section...Math.Section (recentered test scores for tests taken on or after April 1st 1995)
summary(factor(clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section...Math.Section))
clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section...Math.Section[is.na(clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section...Math.Section)] <- "Not Submitted"
clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section...Math.Section <- factor(clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section...Math.Section, order =TRUE, levels = c("Not Submitted", "970", "1020", "1030", "1040", "1050", "1060", "1070", "1080", "1090", "1100", "1110", "1120", "1130", "1140", "1150", "1160", "1170", "1180", "1190", "1200", "1210", "1220","1230", "1240", "1250", "1260", "1270", "1280", "1290", "1300","1310", "1320", "1330", "1340", "1350", "1360", "1370", "1380", "1390", "1400", "1410", "1420", "1430", "1440", "1450", "1460","1470", "1480", "1490", "1500", "1510", "1520", "1530", "1540","1550", "1560", "1570", "1580", "1590", "1600"))
summary(clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section...Math.Section)

#Column 55 - Permanent.Geomarket
summary(factor(clean_data$Permanent.Geomarket))
which(is.na(clean_data$Permanent.Geomarket))
clean_data$Permanent.Geomarket[is.na(clean_data$Permanent.Geomarket)] <- "(Other)"
summary(clean_data$Permanent.Geomarket)
clean_data$Permanent.Geomarket <- factor(clean_data$Permanent.Geomarket)
summary(clean_data$Permanent.Geomarket)

#Column 56 - Citizenship.Status
summary(factor(clean_data$Citizenship.Status))
clean_data$Citizenship.Status <- factor(clean_data$Citizenship.Status)
clean_data$Permanent.Geomarket <- factor(clean_data$Permanent.Geomarket)

#Column 58 - Intend.to.Apply.for.Financial.Aid.
summary(factor(clean_data$Intend.to.Apply.for.Financial.Aid.))
clean_data$Intend.to.Apply.for.Financial.Aid.[is.na(clean_data$Intend.to.Apply.for.Financial.Aid.)] <- 1
clean_data$Intend.to.Apply.for.Financial.Aid. <- factor(clean_data$Intend.to.Apply.for.Financial.Aid.)
summary(clean_data$Intend.to.Apply.for.Financial.Aid.)

#Column 59 - Merit.Award
summary(factor(clean_data$Merit.Award))
#Combine no merit 
clean_data$Merit.Award[clean_data$Merit.Award == "Z0"] <- "No Merit"
clean_data$Merit.Award[clean_data$Merit.Award == "I0"] <- "No Merit"
#Combine 9k options
clean_data$Merit.Award[clean_data$Merit.Award == "TT9"] <- "9K"
clean_data$Merit.Award[clean_data$Merit.Award == "I9"] <- "9K"
#Combine 12.5k options
clean_data$Merit.Award[clean_data$Merit.Award == "D12.5"] <- "12.5K"
clean_data$Merit.Award[clean_data$Merit.Award == "I12.5"] <- "12.5K"
clean_data$Merit.Award[clean_data$Merit.Award == "TT125"] <- "12.5K"
#Combine 17k
clean_data$Merit.Award[clean_data$Merit.Award == "P17"] <- "17K"
clean_data$Merit.Award[clean_data$Merit.Award == "I17"] <- "17K"
#Combine 21k options
clean_data$Merit.Award[clean_data$Merit.Award == "T21"] <- "21K"
clean_data$Merit.Award[clean_data$Merit.Award == "I21"] <- "21K"
#Combine 24k options
clean_data$Merit.Award[clean_data$Merit.Award == "M24"] <- "24K"
clean_data$Merit.Award[clean_data$Merit.Award == "I24"] <- "24K"
#Combine Full Ride 
clean_data$Merit.Award[clean_data$Merit.Award == "SEM"] <- "Full Ride"
clean_data$Merit.Award[clean_data$Merit.Award == "TTS"] <- "Full Ride"
#Full Ride (Faculty/Exchange)
clean_data$Merit.Award[clean_data$Merit.Award == "X0"] <- "Full Ride"
clean_data$Merit.Award[clean_data$Merit.Award == "Y0"] <- "Full Ride"

#18
clean_data$Merit.Award[clean_data$Merit.Award == "D18"] <- "18K"
clean_data$Merit.Award[clean_data$Merit.Award == "I18"] <- "18K"
clean_data$Merit.Award[clean_data$Merit.Award == "P18"] <- "18K"

#20
clean_data$Merit.Award[clean_data$Merit.Award == "D20"] <- "20K"
clean_data$Merit.Award[clean_data$Merit.Award == "I20"] <- "20K"

#10
clean_data$Merit.Award[clean_data$Merit.Award == "TT10"] <- "10K"
clean_data$Merit.Award[clean_data$Merit.Award == "I10"] <- "10K"

#26
clean_data$Merit.Award[clean_data$Merit.Award == "M26"] <- "26K"
clean_data$Merit.Award[clean_data$Merit.Award == "I26"] <- "26K"

#12
clean_data$Merit.Award[clean_data$Merit.Award == "TT12"] <- "12K"
clean_data$Merit.Award[clean_data$Merit.Award == "I12"] <- "12K"

#23
clean_data$Merit.Award[clean_data$Merit.Award == "T23"] <- "23K"
clean_data$Merit.Award[clean_data$Merit.Award == "I23"] <- "23K"
clean_data$Merit.Award[clean_data$Merit.Award == "P23"] <- "23K"

#22
clean_data$Merit.Award[clean_data$Merit.Award == "T22"] <- "22K"
clean_data$Merit.Award[clean_data$Merit.Award == "I22"] <- "22K"

#25
clean_data$Merit.Award[clean_data$Merit.Award == "T25"] <- "25K"
clean_data$Merit.Award[clean_data$Merit.Award == "I25"] <- "25K"
clean_data$Merit.Award[clean_data$Merit.Award == "M25"] <- "25K"

#27
clean_data$Merit.Award[clean_data$Merit.Award == "I27"] <- "27K"
clean_data$Merit.Award[clean_data$Merit.Award == "M27"] <- "27K"

#30
clean_data$Merit.Award[clean_data$Merit.Award == "I30"] <- "30K"
clean_data$Merit.Award[clean_data$Merit.Award == "M30"] <- "30K"
summary(factor(clean_data$Merit.Award))

clean_data$Merit.Award[clean_data$Merit.Award == "No Merit"] <- 0
clean_data$Merit.Award[clean_data$Merit.Award == "I5"] <- 5000
clean_data$Merit.Award[clean_data$Merit.Award == "I7.5"] <- 7500
clean_data$Merit.Award[clean_data$Merit.Award == "9K"] <- 9000
clean_data$Merit.Award[clean_data$Merit.Award == "10K"] <- 10000
clean_data$Merit.Award[clean_data$Merit.Award == "12K"] <- 12000
clean_data$Merit.Award[clean_data$Merit.Award == "12.5K"] <- 12500
clean_data$Merit.Award[clean_data$Merit.Award == "I15"] <- 15000
clean_data$Merit.Award[clean_data$Merit.Award == "17K"] <- 17000
clean_data$Merit.Award[clean_data$Merit.Award == "18K"] <- 18000
clean_data$Merit.Award[clean_data$Merit.Award == "I19"] <- 19000
clean_data$Merit.Award[clean_data$Merit.Award == "20K"] <- 20000
clean_data$Merit.Award[clean_data$Merit.Award == "21K"] <- 21000
clean_data$Merit.Award[clean_data$Merit.Award == "22K"] <- 22000
clean_data$Merit.Award[clean_data$Merit.Award == "23K"] <- 23000
clean_data$Merit.Award[clean_data$Merit.Award == "24K"] <- 24000
clean_data$Merit.Award[clean_data$Merit.Award == "25K"] <- 25000
clean_data$Merit.Award[clean_data$Merit.Award == "26K"] <- 26000
clean_data$Merit.Award[clean_data$Merit.Award == "27K"] <- 27000
clean_data$Merit.Award[clean_data$Merit.Award == "I28"] <- 28000
clean_data$Merit.Award[clean_data$Merit.Award == "30K"] <- 30000
clean_data$Merit.Award[clean_data$Merit.Award == "I32"] <- 32000
clean_data$Merit.Award[clean_data$Merit.Award == "I33"] <- 33000
clean_data$Merit.Award[clean_data$Merit.Award == "I35"] <- 35000
clean_data$Merit.Award[clean_data$Merit.Award == "I38"] <- 38000
clean_data$Merit.Award[clean_data$Merit.Award == "I40"] <- 40000
clean_data$Merit.Award[clean_data$Merit.Award == "I43"] <- 43000
clean_data$Merit.Award[clean_data$Merit.Award == "I45"] <- 45000
clean_data$Merit.Award[clean_data$Merit.Award == "I50"] <- 50000
clean_data$Merit.Award[clean_data$Merit.Award == "I52"] <- 52000
clean_data$Merit.Award[clean_data$Merit.Award == "Full Ride"] <- 65798
summary(clean_data$Merit.Award)
unique(clean_data$Merit.Award)
clean_data$Merit.Award <- as.numeric(clean_data$Merit.Award)

#Column 60 - SAT.Concordance.Score..of.SAT.R. (Translation of old SAT scores to new SAT scores. SAT test changed in 2018)
summary(factor(clean_data$SAT.Concordance.Score..of.SAT.R.))
clean_data$SAT.Concordance.Score..of.SAT.R.[is.na(clean_data$SAT.Concordance.Score..of.SAT.R.)] <- "Not Submitted"
clean_data$SAT.Concordance.Score..of.SAT.R.<- factor(clean_data$SAT.Concordance.Score..of.SAT.R., order =TRUE, levels = c("Not Submitted", "890", "940", "950", "960", "970", "980", "990", "1000", "1010", "1020", "1030", "1040", "1060", "1070", "1080", "1090", "1100", "1110", "1120", "1130", "1140", "1150", "1160", "1170", "1180", "1190", "1200", "1210", "1220","1230", "1250", "1260", "1270", "1280", "1290", "1300","1310", "1320", "1330", "1340", "1350", "1370", "1380", "1390", "1400", "1410", "1420", "1430", "1450", "1460","1470", "1490", "1500", "1510", "1530", "1540", "1560", "1570", "1580", "1600"))
summary(clean_data$SAT.Concordance.Score..of.SAT.R.)

#Column 61 - ACT.Concordance.Score..of.SAT.R. (ACT equivalent of SAT R... recentered ACT)
summary(factor(clean_data$ACT.Concordance.Score..of.SAT.R.))
clean_data$ACT.Concordance.Score..of.SAT.R.[is.na(clean_data$ACT.Concordance.Score..of.SAT.R.)] <- "No Submission"
clean_data$ACT.Concordance.Score..of.SAT.R.<- factor(clean_data$ACT.Concordance.Score..of.SAT.R., order =TRUE, levels = c("No Submission", "15","17", "18", "19", "20", "21", "22", "23","24", "25", "26", "27", "28", "29", "30", "31", "32", "33","34", "35", "36"))

#Column 62 - ACT.Concordance.Score..of.SAT. (ACT equivalent of SAT score without recentering)
summary(factor(clean_data$ACT.Concordance.Score..of.SAT.))
#remove, very few values
clean_data <-subset(clean_data, select = -ACT.Concordance.Score..of.SAT.)

#Column 63 - Test.Optional (fall 2021 admission was test optional. 1 = no send)
summary(factor(clean_data$Test.Optional))
#remove 
clean_data <-subset(clean_data, select = -Test.Optional)

#Column 64 - SAT.I.Critical.Reading
summary(factor(clean_data$SAT.I.Critical.Reading))
1-(14573/15143)
#remove bc non-NA values makes up 3%  of all values 
clean_data <-subset(clean_data, select = -SAT.I.Critical.Reading)

#Column 65 - SAT.I.Math
summary(factor(clean_data$SAT.I.Math))
#remove bc non-NA values makes up 3%  of all values 
clean_data <-subset(clean_data, select = -SAT.I.Math)

#Column 66 - SAT.I.Writing
summary(factor(clean_data$SAT.I.Writing))
#remove bc non-NA values makes up 3%  of all values 
clean_data <-subset(clean_data, select = -SAT.I.Writing)

#Column 67 - SAT.R.Evidence.Based.Reading.and.Writing.Section
summary(factor(clean_data$ SAT.R.Evidence.Based.Reading.and.Writing.Section))
clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section[is.na(clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section)] <- "Not Submitted"
clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section<- factor(clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section, order =TRUE, levels = c("Not Submitted", "450", "460", "480", "490", "500", "510","520", "530", "540", "550", "560", "570", "580", "590", "600", "610","620", "630", "640", "650", "660", "670", "680", "690", "700","710", "720", "730", "740", "750", "760", "770", "780", "790", "800"))
summary(clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section)
clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section[is.na(clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section)] <- "Not Submitted"
summary(clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section)


#Column 68 - SAT.R.Math.Section
summary(factor(clean_data$SAT.R.Math.Section))
clean_data$SAT.R.Math.Section[is.na(clean_data$SAT.R.Math.Section)] <- "Not Submitted"
clean_data$SAT.R.Math.Section<- factor(clean_data$SAT.R.Math.Section, order =TRUE, levels = c("Not Submitted", "450", "460", "480", "490", "500", "510","520", "530", "540", "550", "560", "570", "580", "590", "600", "610","620", "630", "640", "650", "660", "670", "680", "690", "700","710", "720", "730", "740", "750", "760", "770", "780", "790", "800"))
summary(clean_data$SAT.R.Math.Section)

#Column 69 - Decision
skewness(clean_data$Decision)
clean_data$Decision <- as.factor(clean_data$Decision)

#Confirm that data is free of NAs
colnames(is.na(clean_data))
which(colSums(is.na(clean_data))>0)

####################################################################################################

#Create a new column for ACT
clean_data$BestScore.ConvertedACT <- NA

summary(clean_data$ACT.Composite)
summary(clean_data$ACT.Concordance.Score..of.SAT.R.)

clean_data$BestScore.ConvertedACT<- factor(clean_data$BestScore.ConvertedACT, order =TRUE, levels = c("No Submission", "15","17", "18", "19", "20", "21", "22", "23","24", "25", "26", "27", "28", "29", "30", "31", "32", "33","34", "35", "36"))

clean_data$BestScore.ConvertedACT <- pmax(clean_data$ACT.Concordance.Score..of.SAT.R., clean_data$ACT.Composite)
summary(clean_data$BestScore.ConvertedACT)

# Remove ACT.Composite Column and ACT.Concordance.Score..of.SAT.R.
clean_data <-subset(clean_data, select = -ACT.Concordance.Score..of.SAT.R.)
clean_data <-subset(clean_data, select = -ACT.Composite)

# Need to get SAT Concordance.Score.SAT.R converted to ACT
summary(clean_data$SAT.Concordance.Score..of.SAT.R.)
SAT.Concord.Vec <- c("Not Submitted", "890", "940", "950", "960", "970", "980", "990", "1000", "1010", "1020", "1030", "1040", "1060", "1070", "1080", "1090", "1100", "1110", "1120", "1130", "1140", "1150", "1160", "1170", "1180", "1190", "1200", "1210", "1220","1230", "1250", "1260", "1270", "1280", "1290", "1300","1310", "1320", "1330", "1340", "1350", "1370", "1380", "1390", "1400", "1410", "1420", "1430", "1450", "1460","1470", "1490", "1500", "1510", "1530", "1540", "1560", "1570", "1580", "1600")
ACT.Conversion.Vec <- c("Not Submitted", "16", "17", "17", "18", "18", "18", "19", "19", "19", "19", "20", "20", "21", "21", "21", "21", "22", "22", "22", "23", "23", "23", "24", "24", "24", "24", "25", "25", "25","26", "26", "27", "27", "27", "27", "28","28", "28", "29", "29", "29", "30", "30", "31", "31", "31", "32", "32", "33", "33","33", "34", "34", "34", "35", "35", "35", "36", "36", "36")
#Create conversion table
SAT.to.ACT.Conversion <- cbind(SAT.Concord.Vec, ACT.Conversion.Vec)
SAT.to.ACT.Conversion <- as.data.frame(SAT.to.ACT.Conversion)

row.count <- nrow(SAT.to.ACT.Conversion)
row.count

SAT.values <- clean_data[,'SAT.Concordance.Score..of.SAT.R.']
SAT.values

for (i in 1:row.count) {
  search_keyword <- SAT.to.ACT.Conversion[i,1]
  replace_keyword <- SAT.to.ACT.Conversion[i,2]
  
  SAT.values <- sapply(SAT.values, function(x){
    x[x==search_keyword] <- replace_keyword
    return(x)
  })
}
summary(factor(SAT.values))

# Impute SAT.Concordance.Score..of.SAT.R. with ACT versions of those scores
clean_data[,'SAT.Concordance.Score..of.SAT.R.'] <- SAT.values
summary(factor(clean_data$SAT.Concordance.Score..of.SAT.R.))
# Replace not submitted with no submission 
clean_data$SAT.Concordance.Score..of.SAT.R.[clean_data$SAT.Concordance.Score..of.SAT.R.=="Not Submitted"] <- "No Submission"
# Factor
summary(factor(clean_data$SAT.Concordance.Score..of.SAT.R.))
summary(clean_data$BestScore.ConvertedACT)
clean_data$SAT.Concordance.Score..of.SAT.R.<- factor(clean_data$SAT.Concordance.Score..of.SAT.R., order =TRUE, levels = c("No Submission", "15","16","17", "18", "19", "20", "21", "22", "23","24", "25", "26", "27", "28", "29", "30", "31", "32", "33","34", "35", "36"))
clean_data$BestScore.ConvertedACT<- factor(clean_data$BestScore.ConvertedACT, order =TRUE, levels = c("No Submission","15","16","17", "18", "19", "20", "21", "22", "23","24", "25", "26", "27", "28", "29", "30", "31", "32", "33","34", "35", "36"))
summary(clean_data$SAT.Concordance.Score..of.SAT.R.)
summary(clean_data$BestScore.ConvertedACT)


#Ff value in BestScore is "No Submission", then look at SAT.Concordance.Score..of.SAT.R. and impute the "No Submission" value with what is in the SAT.Concordance.Score..of.SAT.R. column. 
library(data.table)
setDT(clean_data)
clean_data[BestScore.ConvertedACT == "No Submission", BestScore.ConvertedACT := SAT.Concordance.Score..of.SAT.R.]
summary(clean_data$BestScore.ConvertedACT)

#######################Stop###############

clean_data$BestScore.ConvertedACT <- with(clean_data, ifelse( BestScore.ConvertedACT == "No Submission",SAT.Concordance.Score..of.SAT.R. , BestScore.ConvertedACT))
summary(factor(clean_data$BestScore.ConvertedACT))
#refactor
clean_data$BestScore.ConvertedACT<- factor(clean_data$BestScore.ConvertedACT, order =TRUE, levels = c("No Submission","6","7","8","9", "10","11","12","13","14", "15","16","17", "18", "19", "20", "21", "22", "23","24", "25", "26", "27", "28", "29", "30", "31", "32", "33","34", "35", "36"))
summary(clean_data$BestScore.ConvertedACT)
clean_data$BestScore.ConvertedACT[is.na(clean_data$BestScore.ConvertedACT)]<- "No Submission"
colnames(clean_data)
#remove SAT.Concordance.Score..of.SAT.R.
clean_data <-subset(clean_data, select = -SAT.Concordance.Score..of.SAT.R.)
#remove SAT.R.Evidence.Based.Reading.and.Writing.Section
clean_data <-subset(clean_data, select = -SAT.R.Evidence.Based.Reading.and.Writing.Section)
#Remove SAT.R.Evidence.Based.Reading.and.Writing.Section...Math.Section
clean_data <-subset(clean_data, select = -SAT.R.Evidence.Based.Reading.and.Writing.Section...Math.Section)
#Remove SAT.R.Math.Section
clean_data <-subset(clean_data, select = -SAT.R.Math.Section)

##################################
#Consider removing individual ACT subject scores if it leads to too much multicollinearity. 

##################
#Split the data
set.seed(123)
Train <- clean_data[1:10000,]
Test <- clean_data[10001:15143,]

#Note many lines are intentionally made into comments so R will not run errors. 
###1. Logistic Regression
#Logistic_model <- glm(Decision ~ ., family = binomial(link = "logit"),
#                 data = Train)
#contrasts(Train$Decision)
# No acceptance is coded 0 and acceptance is coded 1
#summary(Logistic_model)

#try removing permanent geomarket... see if that gets rid of some singularities 
clean_data <-subset(clean_data, select = -Permanent.Geomarket)

#Split the data
#set.seed(123)
#Train <- clean_data[1:10000,]
#Test <- clean_data[10001:15143,]


###1. Logistic Regression
#Logistic_model <- glm(Decision ~ ., family = binomial(link = "logit"),
#    data = Train)
#contrasts(Train$Decision)
#summary(Logistic_model)
#HELPS A LOT TO REMOVE PERM.GEOMARKET. 

###############
#Sport.1.SportNo Sport and Sport.1.Rating.C  are aliased 


################
#Split the data
#set.seed(123)
#Train <- clean_data[1:10000,]
#Test <- clean_data[10001:15143,]

#Logistic_model <- glm(Decision ~. - Sport.1.Sport - Sport.1.Rating, family = binomial(link = "logit"),
#   data = Train)
#contrasts(Train$Decision)
# No acceptance is coded 0 and acceptance is coded 1
#P(y=MM|X) and X
#summary(Logistic_model)

clean_data <-subset(clean_data, select = -Sport.1.Sport)
clean_data <-subset(clean_data, select = -Sport.1.Rating)
clean_data <-subset(clean_data, select = -Academic.Index)

#set.seed(123)
#Train <- clean_data[1:10000,]
#Test <- clean_data[10001:15143,]

#Logistic_model <- glm(Decision ~., family = binomial(link = "logit"),
#    data = Train)
#contrasts(Train$Decision)
# No acceptance is coded 0 and acceptance is coded 1
#P(y=MM|X) and X
#summary(Logistic_model)

#Logistic_prob_train <- predict(Logistic_model, type = "response", Train)
#Logistic_pred_train <- ifelse(Logistic_prob_train > 0.5, "0", "1")
#Logistic_conting_train <- table(Logistic_pred_train, Train$Decision, 
#     dnn = c("Predicted", "Actual"))
##Logistic_conting_train
#Logistic_cm_train <- confusionMatrix(Logistic_conting_train)
#Logistic_cm_train

#Logistic_model <- glm(Decision ~. - Entry.Term..Application., family = binomial(link = "logit"),
#    data = Train)
#vif(Logistic_model)

#Logistic_model <- glm(Decision ~. - Entry.Term..Application. - Academic.Interest.2, family = binomial(link = "logit"),
#   data = Train)
#vif(Logistic_model)

#Logistic_model <- glm(Decision ~. - Entry.Term..Application. - Academic.Interest.2 - First_Source.Origin.First.Source.Summary, family = binomial(link = "logit"),
#                    data = Train)
#vif(Logistic_model)
#summary(Logistic_model)

#Logistic_prob_train <- predict(Logistic_model, type = "response", Train)
#Logistic_pred_train <- ifelse(Logistic_prob_train > 0.5, "1", "0")
#Logistic_conting_train <- table(Logistic_pred_train, Train$Decision, 
#                               dnn = c("Predicted", "Actual"))
#Logistic_conting_train
#Logistic_cm_train <- confusionMatrix(Logistic_conting_train)
#Logistic_cm_train
#Kappa .4929

clean_data <-subset(clean_data, select = -Entry.Term..Application.)
clean_data <-subset(clean_data, select = -Academic.Interest.2)
clean_data <-subset(clean_data, select = -First_Source.Origin.First.Source.Summary)

set.seed(123)
Train <- clean_data[1:10000,]
Test <- clean_data[10001:15143,]


Logistic_model <- glm(Decision ~., family = binomial(link = "logit"),
                      data = Train)
contrasts(Train$Decision)
vif(Logistic_model)
summary(Logistic_model)
set.seed(123)
Logistic_prob_train <- predict(Logistic_model, type = "response", Train)
Logistic_pred_train <- ifelse(Logistic_prob_train > 0.5, "1", "0")
Logistic_conting_train <- table(Logistic_pred_train, Train$Decision, 
                                dnn = c("Predicted", "Actual"))
Logistic_conting_train
Logistic_cm_train <- confusionMatrix(Logistic_conting_train)
Logistic_cm_train
#Kappa .4922 of train Logistic Regression

set.seed(123)
Logistic_prob_test <- predict(Logistic_model, type = "response", Test)
Logistic_pred_results <- ifelse(Logistic_prob_test > 0.5, "1", "0")

Logistic_conting_test <- table(Logistic_pred_results, Test$Decision, 
                               dnn = c("Predicted", "Actual"))
Logistic_conting_test
Logistic_cm_test <- confusionMatrix(Logistic_conting_test)
#Examine the confusion matrix for the test set.
Logistic_cm_test
Logistic_cm_test$overall["Kappa"] 
#test Kappa = .5301196


##################################
### Simple Tree
###3. Simple Classification Tree
#3.1 Fit a simple/basic classification tree.
set.seed(123)
simple_tree <- tree(Decision ~., Train)
summary(simple_tree)
plot(simple_tree)
text(simple_tree, pretty = 0)
set.seed(123)
simple_tree_pred_train <- predict(simple_tree, Train, type = "class")
simple_tree_conting_train <- table(simple_tree_pred_train, 
                                   Train$Decision, 
                                   dnn = c("Predicted", "Actual"))
simple_tree_conting_train
simple_tree_cm_train <- confusionMatrix(simple_tree_conting_train)
simple_tree_cm_train
simple_tree_cm_train$overall["Kappa"]
#Kappa = 0.18257


#3.3 Classification performance (Kappa) for the test set
set.seed(123)
simple_tree_pred_test <- predict(simple_tree, Test, type = "class")
simple_tree_conting_test <- table(simple_tree_pred_test, 
                                  Test$Decision, 
                                  dnn = c("Predicted", "Actual"))
simple_tree_conting_test
simple_tree_cm_test <- confusionMatrix(simple_tree_conting_test)
simple_tree_cm_test
simple_tree_cm_test$overall["Kappa"]
#Kappa = 0.25166

#############################################################################
###4.Pruning the Tree
set.seed(1)
cv_Tree <- cv.tree(simple_tree, FUN = prune.misclass, K = 10)
cv_Tree$size[which.min(cv_Tree$dev)]
#The output shows that the tree with 7 terminal nodes results in the
#lowest cv error.
prune_tree <- prune.misclass(simple_tree, best = 7)
plot(prune_tree)
text(prune_tree, pretty = 0)

#4.3 Classification performance (Kappa) for the training set
set.seed(123)
prune_tree_pred_train <- predict(prune_tree, Train, type = "class")
prune_tree_conting_train <- table(prune_tree_pred_train, 
                                  Train$Decision, 
                                  dnn = c("Predicted", "Actual"))
prune_tree_conting_train
prune_tree_cm_train <- confusionMatrix(prune_tree_conting_train)
prune_tree_cm_train
prune_tree_cm_train$overall["Kappa"]
#kappa = 0.1825739


#4.4 Classification performance (Kappa) for the test set
set.seed(123)
prune_tree_pred_test <- predict(prune_tree, Test, type = "class")
prune_tree_conting_test <- table(prune_tree_pred_test, 
                                 Test$Decision, 
                                 dnn = c("Predicted", "Actual"))
prune_tree_conting_test
prune_tree_cm_test <- confusionMatrix(prune_tree_conting_test)
prune_tree_cm_test
prune_tree_cm_test$overall["Kappa"]
#kappa = 0.2516601
#Comparing kappa score of the pruned tree with that of the un-pruned tree,
#we observe that a simpler tree (i.e. pruned tree) gives us a Kappa
#that is very similar to the Kappa of a complex tree (i.e. un-pruned tree).
#This also implies that the un-pruned tree is a bit over fitted.
#We should choose the pruned tree over the un-pruned tree because
#first, the former has very decent classification performance,
#second, the former is easier to interpret.

########################################################################
##5. Bagging
#5.1 Create 500 bootstrap datasets to perform bagging
#set.seed to make the random sampling with replacement reproducible.
set.seed(1)
#Remember that for Bagging, mtry = p (the number of predictors in the model).
Tree_Bagging <- randomForest(Decision ~ ., data = Train,
                             ntrees = 500, mtry = 27, replace = TRUE,
                             importance = TRUE)
fit_rf<-randomForest(Decision~.,
                     data=Train, ntrees = 500, mtry = 27, replace = TRUE,
                     importance = TRUE,
                     prOximity=TRUE,
                     na.action=na.roughfix)
fit_rf
fit_rf$err.rate
which.min(fit_rf$err.rate[ , 1])
importance(fit_rf)

#5.2 Classification performance (Kappa) for the training set
set.seed(123)
bag_train_pred <- predict(fit_rf, Train, type = "class")
bag_conting_train <- table(bag_train_pred, Train$Decision, 
                           dnn = c("Predicted", "Actual"))
bag_conting_train
bag_cm_train <- confusionMatrix(bag_conting_train)
bag_cm_train
bag_cm_train$overall["Kappa"]
#Kappa = 1, indicating the bagged tree very strongly
#agrees with reality in training set.


#5.3 Classification performance (Kappa) for the test set
set.seed(123)
bag_test_pred <- predict(fit_rf, Test, type = "class")
bag_conting_test <- table(bag_test_pred, Test$Decision, 
                          dnn = c("Predicted", "Actual"))
bag_conting_test
bag_cm_test <- confusionMatrix(bag_conting_test)
bag_cm_test
bag_cm_test$overall["Kappa"]
#Kappa = 0.5172989
################################################################
###6.Random Forest
#set.seed(123)
#Test_Kappa_RF <- rep(0, 26)
#for(i in 1:26){
# set.seed(1)
# Tree_RF <- randomForest(Decision ~ ., data = Train,
#                         ntrees = 500, mtry = i, replace = TRUE,
#                         importance = TRUE)
# Test_pred_RF <- predict(Tree_RF, Test, type = "class")
# RF_conting_test <- table(Test_pred_RF, Test$Decision, 
#                          dnn = c("Predicted", "Actual"))
# RF_cm_test <- confusionMatrix(RF_conting_test)
# Test_Kappa_RF[i] <- RF_cm_test$overall["Kappa"]
#}
#which.max(Test_Kappa_RF)
#mtry = 6 gives the highest Kappa.
#set.seed(123)
#Test_Kappa_RF[which.max(Test_Kappa_RF)]

#6.2 Classification performance (Kappa) for the training set
#using the optimal mtry
#set.seed(1)
#Tree_RF <- randomForest(Decision ~ ., data = Train,
#                       ntrees = 500, mtry = 6, replace = TRUE,
#                       importance = TRUE)
#rf_train_pred <- predict(Tree_RF, Train, type = "class")
#rf_conting_train <- table(rf_train_pred, Train$Decision, 
#                         dnn = c("Predicted", "Actual"))
#rf_conting_train
#rf_cm_train <- confusionMatrix(rf_conting_train)
#rf_cm_train$overall["Kappa"]
#Kappa = 1


#6.3 Classification performance (Kappa) for the test set
#set.seed(123)
#rf_test_pred <- predict(Tree_RF, Test, type = "class")
#rf_conting_test <- table(rf_test_pred, Test$Decision, 
#                        dnn = c("Predicted", "Actual"))
#rf_conting_test
#rf_cm_test <- confusionMatrix(rf_conting_test)
#rf_cm_test$overall["Kappa"]
#Kappa = 0.5342861, indicating the RF tree very strongly
#agrees with reality in training set.
#############################################################

###7. Boosting

##########
write.csv(Train,"Train.csv")
write.csv(Test,"Test.csv")
Train1 <- read.csv("Train.csv")
Test1 <- read.csv("Test.csv")
Train1 <- Train1[,-1]
Test1 <- Test1[,-1]
Train1 <- Train1 %>% mutate_if(is.character,as.factor)
Test1 <- Test1 %>% mutate_if(is.character,as.factor)
set.seed(1)
Tree_Boosting <- gbm(Decision ~., data = Train1, distribution = "bernoulli",
                     n.trees = 5000, interaction.depth = 1,
                     shrinkage = 0.1)
boost_prob_train <- predict(Tree_Boosting, type = "response", 
                            newdata = Train1)
boost_pred_results_train <- ifelse(boost_prob_train > 0.5, 1, 0)
boost_conting_train <- table(boost_pred_results_train, Train1$Decision, 
                             dnn = c("Predicted", "Actual"))
boost_conting_train
boost_cm_train <- confusionMatrix(boost_conting_train)
boost_cm_train$overall["Kappa"]
set.seed(1)
boost_prob_test <- predict(Tree_Boosting, type = "response", 
                           Test1)
boost_pred_results_test <- ifelse(boost_prob_test > 0.5, 1, 0)
boost_conting_test <- table(boost_pred_results_test, Test1$Decision, 
                            dnn = c("Predicted", "Actual"))
boost_conting_test
boost_cm_test <- confusionMatrix(boost_conting_test)
boost_cm_test$overall["Kappa"]

#interaction.depth1 kappa: 
#train: .5124836
#test: .1290846

#interaction.depth2 kappa: 
#train: .7513118
#test: .08285263

#interaction.depth3 kappa: 
#train: .9212293 
#test: 0.08476144

#interaction.depth4 kappa: 
#train: .9867049 
#test:.0685732 

#interaction.depth5 kappa: 
#train: .9996858 
#test: .05643911 

#interaction.depth6 kappa: 
#train: 1
#test: .09771105 

##############################################################

#KNN
clean_data$Permanent.Country <- as.numeric(clean_data$Permanent.Country)
summary(clean_data$Permanent.Country)
clean_data$Sex <- as.numeric(clean_data$Sex)
clean_data$Ethnicity <- as.numeric(clean_data$Ethnicity)
clean_data$Race <- as.numeric(clean_data$Race)
clean_data$Religion <- as.numeric(clean_data$Religion)
clean_data$Application.Source <- as.numeric(clean_data$Application.Source)
clean_data$Decision.Plan <- as.numeric(clean_data$Decision.Plan)
clean_data$Legacy <- as.numeric(clean_data$Legacy)
clean_data$Athlete <- as.numeric(clean_data$Athlete)
clean_data$Sport.2.Sport <- as.numeric(clean_data$Sport.2.Sport)
clean_data$Sport.3.Sport <- as.numeric(clean_data$Sport.3.Sport)
clean_data$Academic.Interest.1 <- as.numeric(clean_data$Academic.Interest.1)
clean_data$Total.Event.Participation <- as.numeric(clean_data$Total.Event.Participation)
clean_data$Count.of.Campus.Visits <- as.numeric(clean_data$Count.of.Campus.Visits)
clean_data$ACT.Writing <- as.numeric(clean_data$ACT.Writing)
clean_data$Citizenship.Status <- as.numeric(clean_data$Citizenship.Status)
clean_data$Intend.to.Apply.for.Financial.Aid. <- as.numeric(clean_data$Intend.to.Apply.for.Financial.Aid.)
clean_data$BestScore.ConvertedACT <- as.numeric(clean_data$BestScore.ConvertedACT)
summary(clean_data$Permanent.Country)

set.seed(123)
Train <- clean_data[1:10000,]
Test <- clean_data[10001:15143,]


Kappa <- rep(0, 100)
for(i in 1:100){
  set.seed(1)
  nn_test <- kNN(Decision ~., Train, Test, k = i)
  nn_conting_test <- table(nn_test, Test$Decision, 
                           dnn = c("Predicted", "Actual"))
  nn_cm_test <- confusionMatrix(nn_conting_test)
  Kappa[i] <- nn_cm_test$overall["Kappa"]
}

which.max(Kappa)
#The best model and Kappa for the training set.
set.seed(1)
nn_best_train <- kNN(Decision ~., Train, Train, 
                     k = which.max(Kappa))
nn_best_conting_train <- table(nn_best_train, Train$Decision, 
                               dnn = c("Predicted", "Actual"))
nn_best_cm_train <- confusionMatrix(nn_best_conting_train)
Kappa_train_KNN <- nn_best_cm_train$overall["Kappa"]
Kappa_train_KNN 
#.482962

#test
set.seed(1)
nn_best_test <- kNN(Decision ~., Train, Test, 
                    k = which.max(Kappa))
nn_best_conting_test <- table(nn_best_test, Test$Decision, 
                              dnn = c("Predicted", "Actual"))
nn_best_cm_test <- confusionMatrix(nn_best_conting_test)
Kappa_test_KNN <- nn_best_cm_test$overall["Kappa"]
Kappa_test_KNN
#.29825





# Classical ML — Master Intelligence

Everything the knowledge base holds about the Classical ML program, in one file, organized around the **unit** (the object the portal keys on). It leads with a per-unit fusion of the three streams — what each session **teaches**, **how** it teaches, and how it is **assessed** — then includes the complete pedagogy-pattern and coding-assignment analyses so no detail is lost. Self-contained; no database needed to read it. Generated 2026-07-15.

Contents: Overview · How the program works · Unit-by-unit intelligence · Curriculum map · Pedagogy patterns · Coding assignment analysis · Signature analogies · Signature code idioms · Two-tier assessment · Glossary · Frameworks & tools · Naming audit · Findings.

## Overview

- **4 courses**, 34 modules, 95 content-bearing units, in global teaching order.
- **74 concepts** taught, **40 teaching patterns** observed, **26 coding assignments** analyzed.

| # | Course | course_id | Modules | Units |
|---|---|---|---|---|
| 1 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | 9 | 31 |
| 2 | Supervised Learning: Regression | `7993c9c1-49e9-4119-b9e3-41d670025951` | 11 | 25 |
| 3 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | 5 | 18 |
| 4 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | 9 | 21 |

**Unit registry — every unit keyed course → course_id → topic → topic_id → unit → unit_id (portal join keys, from the DSML mastersheet):**

| # | Course | course_id | Topic | topic_id | Unit | unit_id |
|---|---|---|---|---|---|---|
| 1 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Getting Started | `4e45e886-d685-4f27-b601-03dc8d4b68de` | Course OverView | `32bb3be6-58b9-4bd1-87d8-bafb64e4b85e` |
| 2 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Getting Started | `4e45e886-d685-4f27-b601-03dc8d4b68de` | Introduction to Machine Learning | `aa399c02-f2b0-4575-96cd-7bcc87a98991` |
| 3 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Getting Started | `4e45e886-d685-4f27-b601-03dc8d4b68de` | Machine Learning Life Cycle | `6361eb29-327f-4213-a32b-09d7a577ec7d` |
| 4 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Getting Started | `4e45e886-d685-4f27-b601-03dc8d4b68de` | Setting Up ML Environment | `f396fdbd-aafa-445a-934e-092380fb769f` |
| 5 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Getting Started | `4e45e886-d685-4f27-b601-03dc8d4b68de` | DA with Pandas Part - 1 | `1307f053-f7ee-448e-9c18-162fb44ac08c` |
| 6 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Getting Started | `4e45e886-d685-4f27-b601-03dc8d4b68de` | DA with Pandas Part - 2 | `16452940-00f8-4608-bd75-c56fa7a7858c` |
| 7 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | k - Nearest Neighbors | `3b5ca95a-9245-4ace-96e1-278aa90c33e5` | KNN | `fafee67c-42f4-4b90-ac46-fd836e3b7094` |
| 8 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | k - Nearest Neighbors | `3b5ca95a-9245-4ace-96e1-278aa90c33e5` | KNN Implementation with Scikit-Learn | `42c69476-3676-44ef-88b6-2b7df95cf0e9` |
| 9 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | k - Nearest Neighbors | `3b5ca95a-9245-4ace-96e1-278aa90c33e5` | KNN Advanced | `f207f78c-6409-430e-87ec-f274c33c8236` |
| 10 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | SVM | `0013c21f-585e-49e8-9492-c49927e9d3c6` | SVM | `f846e8e5-7e4a-47e9-89fa-1ae5764f1f0a` |
| 11 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | SVM | `0013c21f-585e-49e8-9492-c49927e9d3c6` | SVM Implementation | `00104cc8-84e3-4e5a-91ba-c5bdfa4f9cbe` |
| 12 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Naive Bayes | `12c1b953-6f40-400e-89cd-3f7961e32255` | Naive Bayes Part 1 | `f7bb0cf4-7ade-43ca-8a20-2740a827d0f0` |
| 13 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Naive Bayes | `12c1b953-6f40-400e-89cd-3f7961e32255` | Naive Bayes Part 2 | `e81901e9-d64b-4254-9584-0961488cf960` |
| 14 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Naive Bayes | `12c1b953-6f40-400e-89cd-3f7961e32255` | Naive Bayes Implementation | `568bb641-fc15-4a8b-8251-7b79b83a7a95` |
| 15 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Evaluation Metrics | `c36fe709-4293-47ef-99a6-4507e97855e0` | Evaluation Metrics | `03d78eba-b2d6-4d87-9cc4-4252d963ac11` |
| 16 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Evaluation Metrics | `c36fe709-4293-47ef-99a6-4507e97855e0` | Evaluation Metrics Implementation | `e9621c82-786f-4134-bea2-b919e244f567` |
| 17 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Bias and Variance | `4a796a5e-0edd-4831-a543-b610b9638b13` | Bias and Variance | `a609d3f2-bd18-405a-a9ab-9bb2fdc1233c` |
| 18 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Decision Tree | `9aeb3074-96fb-4d41-8b4f-46a31bc0f577` | Decision Tree Part 1 | `fd4cffb4-e8e2-44f9-99ef-601eebb976b8` |
| 19 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Decision Tree | `9aeb3074-96fb-4d41-8b4f-46a31bc0f577` | Decision Tree Part 2 | `6adab306-24cf-4efa-b695-74b3444697d9` |
| 20 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Decision Tree | `9aeb3074-96fb-4d41-8b4f-46a31bc0f577` | Decision Tree Implementation Part 1 | `3ea7aa51-1f58-4ebc-a5c6-82c94ff06461` |
| 21 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Decision Tree | `9aeb3074-96fb-4d41-8b4f-46a31bc0f577` | Decision Tree Implementation Part 2 | `6aef2838-d3d4-4742-9ebc-ab8f2fd649f2` |
| 22 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Hyper Parameter | `6cd04044-ed6e-424e-94ee-ecafb49ac68c` | HyperParameter Tuning Part1 | `e1c2f69c-dcd2-4830-b1ab-35ddc58c05d2` |
| 23 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Hyper Parameter | `6cd04044-ed6e-424e-94ee-ecafb49ac68c` | HyperParameter Tuning Part2 | `c872eb3f-fedd-4806-8b22-52d7e5aace4a` |
| 24 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Capstone Project  (Loan Approval Prediction) | `20f23bd0-a447-4b2d-b2b6-be386c229fb2` | Problem Statement | `54f49af2-db7e-40e6-bcdc-76edf0e5f24a` |
| 25 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Capstone Project  (Loan Approval Prediction) | `20f23bd0-a447-4b2d-b2b6-be386c229fb2` | EDA | `2c3a6ee0-d8a5-426f-890b-5021c6e45604` |
| 26 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Capstone Project  (Loan Approval Prediction) | `20f23bd0-a447-4b2d-b2b6-be386c229fb2` | Missing Values and Outliers Treatment | `05bc9d82-9bd3-4551-8ee6-3f60eaa01b98` |
| 27 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Capstone Project  (Loan Approval Prediction) | `20f23bd0-a447-4b2d-b2b6-be386c229fb2` | Feature Engineering - 1 | `65f3e71b-4e79-4a30-8872-1800ef418038` |
| 28 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Capstone Project  (Loan Approval Prediction) | `20f23bd0-a447-4b2d-b2b6-be386c229fb2` | Evaluation Metrics | `4b26cfde-475d-4e35-a781-fb2927608f66` |
| 29 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Capstone Project  (Loan Approval Prediction) | `20f23bd0-a447-4b2d-b2b6-be386c229fb2` | Build Base ML model | `8782b223-5efb-4e91-b62a-90843a3402f3` |
| 30 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Capstone Project  (Loan Approval Prediction) | `20f23bd0-a447-4b2d-b2b6-be386c229fb2` | Feature Engineering - 2 | `3251fa34-213a-4b8b-83d9-c8cfe19a7315` |
| 31 | Introdution to ML and Classification Algorithms | `919bb576-966a-48df-a1e0-b8b071fc7f69` | Capstone Project  (Loan Approval Prediction) | `20f23bd0-a447-4b2d-b2b6-be386c229fb2` | Build ML model and Conclusion | `1e477106-0cb0-4cca-adb5-39295906d2d2` |
| 32 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Simple Linear Regression | `d69961f7-6423-4be1-97b2-e497ce48206b` | Simple Linear Regression | `b3e07b06-c67d-424d-bb81-1890ed49b5fc` |
| 33 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Simple Linear Regression | `d69961f7-6423-4be1-97b2-e497ce48206b` | SLR Implementation | `e81ca31c-2b68-413e-b832-cb418922f7f8` |
| 34 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Multiple Linear Regression | `ff741a2e-89a5-4e6c-aba8-1fa636260e33` | Multiple Linear Regression | `a93ab3de-76c1-4241-80d6-12b455feae96` |
| 35 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Polynomial Regression | `de6ddf16-975d-42f4-ad2a-a8814478a824` | Polynomial Regression | `aff13bca-f90f-4af1-a9bb-423affbe022a` |
| 36 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Gradient Descent | `352f8931-da2f-4a21-a654-9792710297e0` | Gradient Descent Part - 1 | `6e95dd9b-3976-46b1-8994-65b7efa85ff4` |
| 37 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Gradient Descent | `352f8931-da2f-4a21-a654-9792710297e0` | Gradient Descent Part - 2 | `47687052-3500-49c9-877f-08180848bad0` |
| 38 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Gradient Descent | `352f8931-da2f-4a21-a654-9792710297e0` | Gradient Descent | `ab850cc2-fa88-4701-8823-df24750d1f2e` |
| 39 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Assumptions of Linear Regression | `4556f92c-fc01-4401-b00a-0a5467ee3bc3` | Assumptions of Linear Regression | `270fe1ce-03c5-4cf2-bac2-ad419a873157` |
| 40 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Assumptions of Linear Regression | `4556f92c-fc01-4401-b00a-0a5467ee3bc3` | Assumptions of Linear Regression Implementation | `edc853f3-d899-4433-b43d-917a9f4e4868` |
| 41 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Regularization | `ec046aac-2c54-4607-a9aa-8f5eb62105f0` | Regularization | `3473b49f-8226-462e-9ecf-1c5fad041005` |
| 42 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Regularization | `ec046aac-2c54-4607-a9aa-8f5eb62105f0` | Regularization Implementation | `65717cd7-b1e0-4eb1-89e1-5e737e3f6b5a` |
| 43 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Logistic Regression | `43eda1e9-b1e4-4224-96b2-dd0e95fc21f1` | Logistic Regression | `2c658051-df41-40b8-a20c-bb1e5574ea9a` |
| 44 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Logistic Regression | `43eda1e9-b1e4-4224-96b2-dd0e95fc21f1` | Logistic Regression Implementation | `e76ab4a2-54ab-4553-81b1-b48d09253634` |
| 45 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | KNN Regression | `e7d63313-b33e-4738-ab88-df3429bad9f5` | KNN Regression | `e77e2a55-eb64-4f0d-a030-2c4ea40f121b` |
| 46 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Support Vector Regression | `af0e3a8c-5558-4ff8-92e6-7411e629ac73` | Support Vector Regression | `abcebe4a-ef8f-4d62-a081-41ea5724eef4` |
| 47 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Support Vector Regression | `af0e3a8c-5558-4ff8-92e6-7411e629ac73` | Support Vector Regression Implementation | `9d214c9d-8e2a-4be7-995e-4499ae289802` |
| 48 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Decision Tree Regression | `81046be0-0708-453e-9812-2bf297e2a884` | Decision Tree Regression | `daf48d8a-ffb0-409c-97e3-4e474c382c71` |
| 49 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Decision Tree Regression | `81046be0-0708-453e-9812-2bf297e2a884` | Decision Tree Regression Implementation | `6bddc7e4-00e5-4823-a654-9aa239178e15` |
| 50 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Capstone Project (Life Expectancy Prediction) | `63b6a671-a072-4a1b-b487-90b0097ce31c` | Problem Statement | `34540853-2fd4-4182-aef7-1cc74b742684` |
| 51 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Capstone Project (Life Expectancy Prediction) | `63b6a671-a072-4a1b-b487-90b0097ce31c` | EDA | `2c8b680d-0d85-49e8-a832-84aff5d03259` |
| 52 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Capstone Project (Life Expectancy Prediction) | `63b6a671-a072-4a1b-b487-90b0097ce31c` | Missing Values and Outliers Treatment | `c580a020-cade-4b9d-bb3d-c1688649bf13` |
| 53 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Capstone Project (Life Expectancy Prediction) | `63b6a671-a072-4a1b-b487-90b0097ce31c` | Feature Engineering - 1 | `ae4d661a-8367-4dd3-b5cc-6c165a5b44e4` |
| 54 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Capstone Project (Life Expectancy Prediction) | `63b6a671-a072-4a1b-b487-90b0097ce31c` | Build Base ML model | `e6acab55-2daa-4cd3-bdeb-413569251eb2` |
| 55 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Capstone Project (Life Expectancy Prediction) | `63b6a671-a072-4a1b-b487-90b0097ce31c` | Feature Engineering - 2 | `5676068b-47c3-4928-a8d0-7ee89d574259` |
| 56 | Supervised Learning | `7993c9c1-49e9-4119-b9e3-41d670025951` | Capstone Project (Life Expectancy Prediction) | `63b6a671-a072-4a1b-b487-90b0097ce31c` | Build Final ML model | `686d0668-9e6e-4065-8dad-a0ce4c57ae3e` |
| 57 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Introduction to Ensemble Algorithms | `2132eae2-7f36-4b92-ad9b-65d3bef4219b` | Introduction to Ensemble Algorithms | `83421c90-225c-4400-8bed-483ee0923400` |
| 58 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Voting and Stacking | `3ab3aab0-9682-43ec-a7b0-1d9bc457ef1a` | Voting | `6cd972fc-53ed-48fb-91f7-31abe15d6be2` |
| 59 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Voting and Stacking | `3ab3aab0-9682-43ec-a7b0-1d9bc457ef1a` | Stacking | `20a2bd75-b0bf-4a19-a930-30add4929ee7` |
| 60 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Bagging & Random Forest | `4641fecc-398c-4d1a-80df-2e5528115ec2` | Bagging | `a2ab5ef7-40bd-4ebd-ae7a-4b6ca8e478ad` |
| 61 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Bagging & Random Forest | `4641fecc-398c-4d1a-80df-2e5528115ec2` | Random Forest | `275f25cb-aff3-426d-95fe-6e61fcc3f6ed` |
| 62 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Bagging & Random Forest | `4641fecc-398c-4d1a-80df-2e5528115ec2` | AUC - ROC Curve | `0dbe0f80-ed0e-4c44-b116-ddffc69fa79e` |
| 63 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Bagging & Random Forest | `4641fecc-398c-4d1a-80df-2e5528115ec2` | Random Forest Implementation | `f78375df-cbf0-40ee-bdae-11b9de49b7d5` |
| 64 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Bagging & Random Forest | `4641fecc-398c-4d1a-80df-2e5528115ec2` | Feature Importance Techniques | `1b09e4fb-1ace-4f14-be9f-239898a2371f` |
| 65 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Boosting | `58dfde4d-2962-442d-8c5f-82446c50a8ac` | Boosting methods and Adaboost | `cdb69b70-df3b-447c-8fd4-7072f37b4d51` |
| 66 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Boosting | `58dfde4d-2962-442d-8c5f-82446c50a8ac` | Gradient Boosting Regression | `7b383f6b-808d-47fb-bec4-1a2bac38f931` |
| 67 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Boosting | `58dfde4d-2962-442d-8c5f-82446c50a8ac` | Gradient Boosting Classification | `bc8ec92e-65cc-4733-bf1d-739956e45b4e` |
| 68 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Boosting | `58dfde4d-2962-442d-8c5f-82446c50a8ac` | XG boost | `68c7802b-b0bb-49c6-b200-78b77a33df48` |
| 69 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Boosting | `58dfde4d-2962-442d-8c5f-82446c50a8ac` | Boosting Implementation - Classification | `2fdde1a8-9fde-4b5c-bf57-fc32637e9285` |
| 70 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Boosting | `58dfde4d-2962-442d-8c5f-82446c50a8ac` | Boosting Implementation - Regression | `7c098fa1-496a-4ccc-85de-456c22714a4b` |
| 71 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Capstone Project (Credit card fraud Detection | `253b9675-48cb-45d1-b506-1f1263c81935` | Problem Statement | `3879b76b-bdaf-4a97-aaf0-e7c53807fe76` |
| 72 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Capstone Project (Credit card fraud Detection | `253b9675-48cb-45d1-b506-1f1263c81935` | EDA | `e577bba1-ac9c-4a9e-a461-2420146857af` |
| 73 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Capstone Project (Credit card fraud Detection | `253b9675-48cb-45d1-b506-1f1263c81935` | Model Building 1 | `2791df44-9109-45bd-8be2-85134104b060` |
| 74 | Ensemble Learning | `453ae28a-0fd5-42cc-94b3-8705d3018d59` | Capstone Project (Credit card fraud Detection | `253b9675-48cb-45d1-b506-1f1263c81935` | Model Building 2 | `bdc0c7a5-0c7b-450b-befb-b07252d8c005` |
| 75 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Introduction to Unsupervised Learning | `4b52ed88-7747-4df0-9815-290f7aee9d98` | Introduction to Unsupervised Learning | `2285a57a-f498-4625-b4a8-87f2509f1065` |
| 76 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | K-Means Clustering | `16e0f965-8ad9-4578-be48-8e03cc5bd50b` | K-Means Clustering | `8229cfaa-2acc-4006-8ee8-61c6b34d20e5` |
| 77 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | K-Means Clustering | `16e0f965-8ad9-4578-be48-8e03cc5bd50b` | K-Means Clustering Part 2 | `1786d611-c0ee-4836-a306-9ce43fa3cd1a` |
| 78 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | K-Means Clustering | `16e0f965-8ad9-4578-be48-8e03cc5bd50b` | K-Means Clustering Part 3 | `dcfec5f5-07f9-4ab3-a1fd-b743e01945fb` |
| 79 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | K-Means Clustering | `16e0f965-8ad9-4578-be48-8e03cc5bd50b` | K-Means implementation | `00ee847c-ac2a-4b0a-9e5d-e5b062d93694` |
| 80 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Hierarchical Clustering | `1d627f49-f4e8-4da5-a979-26c6cb541d7e` | Hierarchical Clustering | `1761a9de-98d0-4927-a348-418a409d4eaf` |
| 81 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Hierarchical Clustering | `1d627f49-f4e8-4da5-a979-26c6cb541d7e` | Hierarchical Clustering Implementation | `f8e15cd5-e68d-4491-b2d4-04bff5b549d7` |
| 82 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | DBSCAN | `bec64ffc-c86b-418e-874a-cb85a2941110` | DBSCAN | `e73d28de-af17-4a12-9ed4-8586b16a16b4` |
| 83 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | DBSCAN | `bec64ffc-c86b-418e-874a-cb85a2941110` | DBSCAN Implementation | `be6580b6-cf6c-4c37-a374-5864884ec76b` |
| 84 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Apriori Algorithm | `cc46b15a-3e7e-4d14-8e90-6062da0beb12` | Apriori Algorithm | `460cd724-a14f-435b-aab6-e78274072abf` |
| 85 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Apriori Algorithm | `cc46b15a-3e7e-4d14-8e90-6062da0beb12` | Apriori Implementation | `c8a64568-31e1-4281-a51d-59360b83e43f` |
| 86 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Eclat | `ec470e1f-1a4a-4092-9037-d0e401e100b1` | Eclat | `8a31c63c-15dc-42a9-8239-b79d30f1167f` |
| 87 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Eclat | `ec470e1f-1a4a-4092-9037-d0e401e100b1` | Eclat Implementation | `242bbc39-d193-4661-809d-eeec1cb7d0d1` |
| 88 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Dimentionality Reduction | `8dfc2555-803c-4527-a19a-cf97a889a2ac` | Dimentionality Reduction | `1af9101f-6118-4ad9-9560-876be2c79978` |
| 89 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Dimentionality Reduction | `797c6965-615d-455e-82e5-7bec7cab7706` | Dimentionality Reduction Part 2 | `d52d4bae-c9cb-403c-b2d8-18269547509a` |
| 90 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Dimentionality Reduction | `797c6965-615d-455e-82e5-7bec7cab7706` | Dimentionality Reduction Part 3 | `c09b62c5-9b59-429b-96d3-a61b7ba17c68` |
| 91 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Dimentionality Reduction | `797c6965-615d-455e-82e5-7bec7cab7706` | Dimentionality Reduction Implementation | `f2cdf3c3-28a2-4d99-a747-7927b735b93a` |
| 92 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Retail Customer Segmemtation | `797c6965-615d-455e-82e5-7bec7cab7706` | Problem Statement | `4d90fab0-e846-4792-9187-20d57e19e521` |
| 93 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Retail Customer Segmemtation | `797c6965-615d-455e-82e5-7bec7cab7706` | EDA | `91a9326b-bb54-4607-a062-550393cf56b3` |
| 94 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Retail Customer Segmemtation | `797c6965-615d-455e-82e5-7bec7cab7706` | Model Building -1 | `5cefb13c-f534-41b1-bf73-d326b7b75992` |
| 95 | Unsupervised Learning | `4f4407db-75de-4209-8333-5d21abfcc064` | Retail Customer Segmemtation | `797c6965-615d-455e-82e5-7bec7cab7706` | Model Building 2 | `c9ad0d51-5fb7-4525-b9a5-7b04cf4dbc54` |

## How the program works

**Content is workflow-first.** Across all four courses the spine is the same load → EDA → clean/preprocess → train → evaluate pipeline, taught on named real-world datasets. Concepts are introduced as steps in that pipeline, and each algorithm is positioned by contrast with a sibling.

**Pedagogy reinforces it with a stable house style.** The most-observed moves are agenda-first framing, definition callouts, real-world motivation with named domains, the workflow skeleton in notebooks, and when-it-fails honesty. Concepts chain, and notebooks pair every output with a written interpretation.

**Assessment is Kaggle-style.** Coding assignments are submission-file tasks checked against a hidden ground truth with a performance threshold. The strongest verify the exact technique or force it structurally; the recurring gap is four single-model classification assignments (KNN, Naive Bayes, Logistic, Hyperparameter Tuning) that are technique-agnostic and lenient.

**Through-line:** content teaches the pipeline, pedagogy drills it, assessment tests it. Concept depth is strong in slides and outcomes, lighter in the classification assignments — the clearest place to tighten alignment. The unit-by-unit section below makes this visible per session.

## Unit-by-unit intelligence

Each entry carries its `unit_id` for portal join, then the three streams fused.

### Introdution to ML and Classification Algorithms

#### Module 1: Getting Started

##### Course OverView
`unit_id: 32bb3be6-58b9-4bd1-87d8-bafb64e4b85e` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 4e45e886-d685-4f27-b601-03dc8d4b68de`

**Teaches:** _none tagged_

_Flow:_ 1) Welcome and course introduction 2) Meet the instructor and their background 3) Course prerequisites: math foundations, programming, no prior ML needed 4) What the course covers: theory, hands-on, coding, assessments, projects 5) Real-world applications across healthcare, finance, marketing, and manufacturing

_Outcomes (depth):_ Describe what the course covers and how it is structured [Introduced]; Identify the prerequisites needed to start the course [Introduced]; Recognize real-world domains where machine learning is applied [Introduced]

_Key takeaways:_ (1) The course builds a foundation in classical machine learning with hands-on practice on real datasets. (2) No prior machine learning experience is required; basic math and programming are helpful. (3) Learning combines theory, coding, assessments, and projects. (4) Machine learning has broad applications across healthcare, finance, marketing, and manufacturing.

**Taught how (1 patterns):** Real-world motivation with named domains

---

##### Introduction to Machine Learning
`unit_id: aa399c02-f2b0-4575-96cd-7bcc87a98991` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 4e45e886-d685-4f27-b601-03dc8d4b68de`

**Teaches:** Association Rule Learning, Classification, Clustering, Dimensionality Reduction (PCA), Machine Learning, Regression (task), Supervised Learning, Unsupervised Learning

_Flow:_ 1) Define machine learning (Arthur Samuel, 1959) 2) Contrast traditional programming with machine learning 3) Introduce supervised and unsupervised learning 4) Supervised learning: features and targets, classification vs regression 5) Unsupervised learning: clustering, association, dimensionality reduction 6) Visual recap of ML algorithm families with example applications

_Outcomes (depth):_ Define machine learning and distinguish it from traditional programming [Explained]; Differentiate supervised from unsupervised learning by whether labels are present [Explained]; Classify a problem as classification or regression based on its output type [Explained]; Identify clustering, association, and dimensionality reduction as unsupervised tasks [Introduced]; Match example applications to the correct ML algorithm family [Explained]

_Key takeaways:_ (1) Machine learning lets computers learn patterns from data rather than being explicitly programmed. (2) Traditional programming maps data plus a program to output, while machine learning derives the program from data and outputs. (3) Supervised learning needs labeled data (features and targets) and splits into classification for discrete outputs and regression for continuous outputs. (4) Unsupervised learning finds structure in unlabeled data through clustering, association, and dimensionality reduction.

**Taught how (4 patterns):** Agenda-first, Key-Takeaways-last framing; Comparison table contrasting methods; Definition callouts for terminology; Real-world motivation with named domains

**Supporting content:** 1 reading, 1 mcq

---

##### Machine Learning Life Cycle
`unit_id: 6361eb29-327f-4213-a32b-09d7a577ec7d` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 4e45e886-d685-4f27-b601-03dc8d4b68de`

**Teaches:** Machine Learning Lifecycle

_Flow:_ 1) Recap machine learning algorithm types 2) Define the problem 3) Collect data 4) Prepare data 5) Train and evaluate the model 6) Deploy and integrate the model 7) Monitor the model and iterate or retrain

_Outcomes (depth):_ List the stages of the machine learning lifecycle in order [Explained]; Explain why the lifecycle iterates through retraining [Explained]; Order the steps from problem definition through deployment and monitoring [Explained]

_Key takeaways:_ (1) The machine learning lifecycle is an ordered sequence from problem definition to monitoring. (2) Data collection and preparation come before model training and evaluation. (3) A deployed model must be monitored and retrained as conditions change, making the cycle iterative.

**Taught how (2 patterns):** Agenda-first, Key-Takeaways-last framing; Recap bridge from prior session

**Supporting content:** 1 reading, 1 mcq

---

##### Setting Up ML Environment
`unit_id: f396fdbd-aafa-445a-934e-092380fb769f` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 4e45e886-d685-4f27-b601-03dc8d4b68de`

**Teaches:** ML Environment Setup

_Flow:_ 1) Recap the machine learning lifecycle 2) Install Anaconda Navigator and Jupyter Notebook 3) Install core libraries: NumPy, Pandas, Matplotlib 4) Create and activate environments 5) Install packages with pip 6) Deactivate and remove environments

_Outcomes (depth):_ Create and manage conda environments, including activation, deactivation, and removal [Explained]; Install Python packages with pip inside an environment [Explained]; Identify the core libraries used for machine learning: NumPy, Pandas, Matplotlib [Introduced]

_Key takeaways:_ (1) Anaconda Navigator and Jupyter Notebook provide the working environment for machine learning. (2) NumPy, Pandas, and Matplotlib are the core libraries installed for the course. (3) Conda environments can be created, activated, deactivated, and removed to keep setups isolated.

**Taught how (2 patterns):** Agenda-first, Key-Takeaways-last framing; Recap bridge from prior session

**Supporting content:** 1 reading

---

##### DA with Pandas Part - 1
`unit_id: 1307f053-f7ee-448e-9c18-162fb44ac08c` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 4e45e886-d685-4f27-b601-03dc8d4b68de`

**Teaches:** Data Analysis with Pandas, Data Cleaning, Data Visualization, Exploratory Data Analysis (EDA)

_Flow:_ 1) Introduce the Zomato restaurant dataset and the analysis objective 2) Load the data and inspect structure with info and head 3) Drop unnecessary columns and rename columns for readability 4) Remove NaN values 5) Clean individual columns using regex and type conversion for rating and cost 6) Drop duplicate rows and reset the index 7) Save the cleaned dataset 8) Visualize insights with Seaborn and Matplotlib

_Outcomes (depth):_ Clean a raw dataset by dropping columns, renaming, and removing NaN values and duplicates [Applied]; Convert object columns to numeric types using string operations with apply and astype [Applied]; Aggregate data with groupby and agg to summarize by category [Applied]; Build visualizations with Seaborn and Matplotlib to explore the data [Applied]; Interpret plots to answer questions about the dataset [Applied]

_Key takeaways:_ (1) Data cleaning covers dropping redundant columns, renaming, handling NaN values, fixing datatypes, and removing duplicates. (2) Regex and lambda or apply transformations turn messy string columns into usable numeric types. (3) groupby with aggregation summarizes data by category, such as median rating per location. (4) Seaborn and Matplotlib plots such as countplot, barplot, kdeplot, and distplot reveal patterns in the cleaned data. (5) Exploratory analysis answers concrete questions about restaurant trends and preferences.

**Taught how (8 patterns):** Agenda-first, Key-Takeaways-last framing; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Problem statement first (business framing); Question-driven exploratory analysis; Real-world motivation with named domains; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments

**Assessed by — EDA Project — session: DA with Pandas Part - 1** (MEDIUM · 15 test cases · style 3)
- Tests: 15 specific aggregations by value (nulls cleaned, avg purchase by gender/age, top-5 products, correlation matrix).
- Alignment: **strong** — exercises the exact pandas/EDA skills, value-by-value; can't shortcut.
- Value: high. Changes: none urgent — this is the template for concept-faithful evaluation.

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

##### DA with Pandas Part - 2
`unit_id: 16452940-00f8-4608-bd75-c56fa7a7858c` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 4e45e886-d685-4f27-b601-03dc8d4b68de`

**Teaches:** Data Analysis with Pandas, Data Cleaning, Data Visualization, Exploratory Data Analysis (EDA)

_Flow:_ 1) Introduce the Zomato restaurant dataset and the analysis objective 2) Load the data and inspect structure with info and head 3) Drop unnecessary columns and rename columns for readability 4) Remove NaN values and clean individual columns with regex and type conversion 5) Drop duplicate rows and save the cleaned dataset 6) Import Seaborn and Matplotlib for visualization 7) Answer analysis questions with count, bar, and point plots 8) Examine distributions and category aggregations with kde, distplot, catplot, and pie charts

_Outcomes (depth):_ Aggregate restaurant data by category using groupby and agg [Applied]; Create count, bar, point, and pie plots to compare categories [Applied]; Plot feature distributions with kdeplot and distplot [Applied]; Filter rows with conditions and isin to focus the analysis [Applied]; Interpret the visualizations to draw conclusions about restaurant trends [Applied]

_Key takeaways:_ (1) Cleaned data enables reliable exploratory data analysis and visualization. (2) groupby combined with agg produces summaries such as total votes and mean rating per restaurant type. (3) Different plot types answer different questions: countplot for frequencies, barplot for comparisons, distplot for distributions. (4) A near-normal distribution of ratings shows most values fall between 3.5 and 4.5. (5) Visual interpretation converts raw counts into business insights about locations, types, and cuisines.

**Taught how (8 patterns):** Agenda-first, Key-Takeaways-last framing; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Problem statement first (business framing); Question-driven exploratory analysis; Real-world motivation with named domains; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments

**Supporting content:** 1 reading, 1 code, 1 mcq

---

#### Module 2: k - Nearest Neighbors

##### KNN
`unit_id: fafee67c-42f4-4b90-ac46-fd836e3b7094` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 3b5ca95a-9245-4ace-96e1-278aa90c33e5`

**Teaches:** Choosing K, Distance Metrics, Euclidean Distance, Hamming Distance, K-Nearest Neighbors (KNN), Manhattan Distance, Minkowski Distance

_Flow:_ 1) Introduce KNN as a simple distance-based classifier 2) Walk through the KNN algorithm steps 3) Discuss how to choose the value of K 4) Introduce the list of distance metrics 5) Euclidean distance 6) Manhattan distance 7) Minkowski distance as a generalization with parameter p 8) Hamming distance 9) Guidance on which metric to use

_Outcomes (depth):_ Explain how KNN classifies a point using its nearest neighbors [Explained]; Describe the ordered steps of the KNN algorithm [Explained]; Apply the Euclidean and Manhattan distance formulas between two points [Explained]; Select an appropriate value of K for a classification problem [Explained]; Choose a distance metric based on feature characteristics [Explained]; Relate Minkowski distance to Euclidean and Manhattan through the parameter p [Explained]

_Key takeaways:_ (1) KNN classifies a new point by taking the majority class among its k nearest neighbors. (2) The value of K affects predictions, and an odd K helps avoid ties in binary classification. (3) Distance metrics such as Euclidean, Manhattan, Minkowski, and Hamming measure closeness in different ways. (4) Minkowski distance generalizes Euclidean (p=2) and Manhattan (p=1) through a parameter p. (5) Metric choice depends on feature types, such as whether features share the same units.

**Taught how (6 patterns):** Agenda-first, Key-Takeaways-last framing; Algorithm stated as an explicit numbered step sequence; Analogy or concrete scenario before formalism; Comparison table contrasting methods; Definition callouts for terminology; When-to-use guidance with concrete triggers

**Assessed by — KNN — session: KNN (Breast Cancer)** (MEDIUM · accuracy > 65% · style 6)
- Tests: build any KNN, output `submission_df`, beat 65% (dataset clears ~95%).
- Alignment: **weak** — none of the taught concepts exercised; defaults pass.
- Value: workflow only. Changes: raise to ~90%, require K exploration or a scaled pipeline, verify a KNN estimator.

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### KNN Implementation with Scikit-Learn
`unit_id: 42c69476-3676-44ef-88b6-2b7df95cf0e9` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 3b5ca95a-9245-4ace-96e1-278aa90c33e5`

**Teaches:** Choosing K, Data Visualization, K-Nearest Neighbors (KNN)

_Flow:_ 1) Define the Iris flower classification problem 2) Load the Iris dataset and inspect its shape and columns 3) Check class balance with value_counts 4) Perform EDA with 2D scatter plots, FacetGrid, and pairplot 5) Split the data into features and target 6) Build and fit a KNeighborsClassifier with scikit-learn 7) Predict a new sample and evaluate with score 8) Loop over K values and plot accuracy to choose an optimal K 9) Review the pros and cons of KNN

_Outcomes (depth):_ Perform EDA on a dataset using scatter plots and pairplots [Applied]; Build and fit a KNN classifier with scikit-learn [Applied]; Predict the class of a new sample and evaluate accuracy with score [Applied]; Select an optimal K by plotting accuracy across K values [Applied]; Summarize the pros and cons of KNN [Explained]

_Key takeaways:_ (1) Exploratory data analysis with scatter plots and pairplots reveals which features best separate classes. (2) scikit-learn's KNeighborsClassifier fits KNN with fit and predicts with predict. (3) Iterating over K values and plotting accuracy guides selection of the optimal K. (4) KNN is simple and has no training phase but is computationally and memory intensive. (5) KNN performance degrades with irrelevant features and is sensitive to the choice of K.

**Taught how (9 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Problem statement first (business framing); Real-world motivation with named domains; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

##### KNN Advanced
`unit_id: f207f78c-6409-430e-87ec-f274c33c8236` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 3b5ca95a-9245-4ace-96e1-278aa90c33e5`

**Teaches:** Feature Scaling, Normalization (Min-Max Scaling), Probabilistic KNN, Standardization (Z-score Scaling), Weighted KNN

_Flow:_ 1) Recap the KNN algorithm 2) Introduce Weighted KNN where closer neighbors have more influence 3) Break down the Weighted KNN formula with weights as inverse distance 4) Introduce Probabilistic KNN using neighbor proportions per class 5) Explain why feature scaling matters for distance-based methods 6) Min-Max scaling (normalization) with a worked example 7) Standardization (Z-score scaling) with worked mean and standard deviation 8) Guidance on when to use each scaling method

_Outcomes (depth):_ Compute neighbor weights as the inverse of distance in Weighted KNN [Applied]; Calculate class probabilities using Probabilistic KNN [Applied]; Explain why feature scaling is necessary for distance-based models [Explained]; Apply Min-Max scaling to rescale features to a [0,1] range [Applied]; Apply standardization using the feature mean and standard deviation [Applied]; Choose between Min-Max scaling and standardization for a given scenario [Explained]

_Key takeaways:_ (1) Weighted KNN gives closer neighbors more influence, typically weighting by the inverse of distance. (2) Probabilistic KNN outputs class probabilities from the proportion of neighbors belonging to each class. (3) Feature scaling prevents features with larger numerical values from dominating distance calculations. (4) Min-Max scaling rescales features to a range such as [0,1], while standardization centers data to mean 0 and standard deviation 1. (5) Choice of scaling depends on whether a bounded range or a Gaussian-centered distribution is needed.

**Taught how (11 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Paired theory-plus-implementation with live-demo cues; Preprocessing justified by the algorithm's needs; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; When-to-use guidance with concrete triggers; Worked example computed step by step

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

#### Module 3: SVM

##### SVM
`unit_id: f846e8e5-7e4a-47e9-89fa-1ae5764f1f0a` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 0013c21f-585e-49e8-9492-c49927e9d3c6`

**Teaches:** Data Visualization, Hard vs Soft Margin, Hyperplane and Margin, Kernel Trick, Kernels (Linear / RBF / Polynomial), Label Encoding, Support Vector Machines (SVM), Support Vectors, Train-Test Split

_Flow:_ 1) What an SVM is: a supervised classifier that finds the best separating hyperplane 2) Intuition: many lines can separate the classes - pick the one with the widest margin 3) Maximal-margin (hard-margin) classifier and support vectors 4) Soft margin (Support Vector Classifier) and the role of C in tolerating misclassification 5) Kernels for non-linear separation: linear, polynomial, RBF/Gaussian 6) SVM for multiclass classification 7) Implementation: mushroom edible-vs-poisonous classification (encode, train an SVC)

_Outcomes (depth):_ Explain how an SVM chooses its decision boundary by maximising the margin [Explained]; Distinguish hard-margin from soft-margin classification and the effect of C [Deep-dive]; Explain how kernels enable non-linear classification [Explained]; Train an SVM classifier on an encoded dataset [Applied]

_Key takeaways:_ (1) An SVM separates classes with the maximum-margin hyperplane; the closest points that define it are the support vectors. (2) A hard margin demands perfect separation; a soft margin (parameter C) tolerates some errors for better generalisation. (3) Kernels map data into a higher dimension so non-linearly separable data becomes separable (linear, polynomial, RBF).

**Taught how (15 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Flag optional or simplified mathematical depth; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Paired theory-plus-implementation with live-demo cues; Problem statement first (business framing); Real-world motivation with named domains; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers

**Assessed by — SVM — session: SVM (Apple Quality)** (MEDIUM · accuracy ≥ 0.80 + macro precision ≥ 0.82 · style 5/6)
- Tests: predictions array + dual metric (accuracy + macro precision). Does not verify an SVC or probe kernel/margin.
- Alignment: partial — strong dual threshold (rewards class balance), but the SVM-specific concepts aren't touched.
- Value: good (dual metric). Changes: fix the 0.82/0.80 display-vs-assert mismatch; consider verifying an SVC.

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

##### SVM Implementation
`unit_id: 00104cc8-84e3-4e5a-91ba-c5bdfa4f9cbe` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 0013c21f-585e-49e8-9492-c49927e9d3c6`

**Teaches:** Label Encoding, Support Vector Machines (SVM), Train-Test Split

_Flow:_ 1) Define the classification problem and load the dataset 2) Perform exploratory data analysis 3) Encode categorical values with label encoding 4) Split the data into train and test sets 5) Build the SVM model with scikit-learn 6) Evaluate the model and predict new values 7) Review the pros and cons of SVM

_Outcomes (depth):_ Apply label encoding to convert categorical values into integers [Applied]; Outline the end-to-end SVM implementation workflow from problem definition to evaluation [Explained]; Describe why SVM requires feature scaling [Explained]; Recognize that the kernel trick lets SVM capture non-linear relationships [Introduced]; Weigh the strengths and weaknesses of SVM for a classification task [Explained]

_Key takeaways:_ (1) The SVM workflow moves from problem definition and loading through EDA, train-test split, model building, and evaluation. (2) Label encoding maps categorical values to integers so a model can consume them. (3) SVM can model non-linear relationships through the kernel trick and is sensitive to the scale of input features. (4) SVM is effective in high dimensions and memory efficient because it relies only on support vectors, but it is computationally intensive and depends heavily on kernel choice. (5) SVM struggles with noisy, overlapping classes and produces models that are hard to interpret.

**Taught how (4 patterns):** Agenda-first, Key-Takeaways-last framing; ML workflow skeleton in implementation notebooks; Real-world motivation with named domains; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

#### Module 4: Naive Bayes

##### Naive Bayes Part 1
`unit_id: f7bb0cf4-7ade-43ca-8a20-2740a827d0f0` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 12c1b953-6f40-400e-89cd-3f7961e32255`

**Teaches:** Naive Bayes

_Flow:_ 1) Introduce Naive Bayes and its use in text classification 2) State the naive conditional independence assumption 3) Present Bayes theorem and explain each term 4) Work the colored-objects example through prior, likelihood, and posterior 5) Compare KNN with Naive Bayes 6) Apply Naive Bayes to a sports text classification example 7) Handle zero counts with Laplace smoothing 8) Classify by choosing the highest posterior (MAP)

_Outcomes (depth):_ Apply Bayes theorem to compute posterior probabilities from priors and likelihoods [Applied]; Classify a text sample by multiplying per-word likelihoods with the class prior [Applied]; Apply Laplace smoothing to avoid zero-frequency probabilities [Applied]; Explain why the independence assumption simplifies computation and where it breaks down [Explained]; Compare KNN and Naive Bayes across assumptions, computation, and output [Deep-dive]

_Key takeaways:_ (1) Naive Bayes is a probabilistic classifier built on Bayes theorem and is widely used for high-dimensional text tasks. (2) The naive assumption treats features as conditionally independent given the class, which simplifies calculation but can miss word combinations. (3) Bayes theorem combines a prior probability and a likelihood to yield a posterior used for classification. (4) For text, word order is ignored and each word contributes an independent likelihood term. (5) Laplace smoothing adds one to counts so an unseen word does not force a probability of zero.

**Taught how (11 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Real-world motivation with named domains; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; When-it-fails honesty / limitations stated; Worked example computed step by step

**Supporting content:** 1 in_class_quiz, 1 mcq

---

##### Naive Bayes Part 2
`unit_id: e81901e9-d64b-4254-9584-0961488cf960` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 12c1b953-6f40-400e-89cd-3f7961e32255`

**Teaches:** Naive Bayes

_Flow:_ 1) Recap Bayes theorem 2) Introduce the three Naive Bayes variants 3) Explain Gaussian Naive Bayes for continuous features 4) Explain Multinomial Naive Bayes for count features 5) Explain Bernoulli Naive Bayes for binary features 6) Review usage notes on normality, Laplace correction, and correlated features

_Outcomes (depth):_ Distinguish Gaussian, Multinomial, and Bernoulli Naive Bayes by feature type [Explained]; Select the appropriate Naive Bayes variant for a given dataset [Explained]; Explain when Laplace correction is needed in Multinomial Naive Bayes [Explained]; Recognize that continuous features may need transformation to meet the normality assumption [Explained]; Identify why highly correlated features should be removed [Introduced]

_Key takeaways:_ (1) Gaussian Naive Bayes assumes continuous, real-valued features follow a normal distribution. (2) Multinomial Naive Bayes models discrete counts such as word frequencies and is common in text classification. (3) Bernoulli Naive Bayes uses binary features that indicate the presence or absence of a term. (4) The right variant depends on whether the features are continuous, counts, or binary. (5) Practical notes: transform non-normal continuous features, apply Laplace correction for zero counts, and drop highly correlated features.

**Taught how (6 patterns):** Agenda-first, Key-Takeaways-last framing; Preprocessing justified by the algorithm's needs; Real-world motivation with named domains; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; When-to-use guidance with concrete triggers

**Assessed by — Naive Bayes — session: Naive Bayes Part 2 (Adult Income)** (MEDIUM · accuracy > 75% · style 6)
- Tests: output `submission_df`, beat 75%. No NB model verified.
- Alignment: **weak** — the session's core (variant choice) is untested; a RandomForest passes.
- Value: workflow only. Changes: assert a `sklearn.naive_bayes` estimator; use a mixed-feature dataset that rewards correct variant.

**Supporting content:** 1 reading, 1 in_class_quiz

---

##### Naive Bayes Implementation
`unit_id: 568bb641-fc15-4a8b-8251-7b79b83a7a95` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 12c1b953-6f40-400e-89cd-3f7961e32255`

**Teaches:** Naive Bayes, Train-Test Split

_Flow:_ 1) Define the spam-or-ham classification problem 2) Load the dataset and perform basic EDA 3) Convert text to count features with CountVectorizer 4) Inspect the vocabulary and the sparse count representation 5) Encode the target as binary and vectorize messages with n-grams 6) Split the data into train and test sets 7) Train a Gaussian Naive Bayes model and score it 8) Compare Naive Bayes with KNN and SVM 9) Predict on new messages and observe a misclassification

_Outcomes (depth):_ Transform raw text into count features using CountVectorizer [Applied]; Build and score a Gaussian Naive Bayes classifier with scikit-learn [Applied]; Compare training and prediction speed across Naive Bayes, KNN, and SVM [Applied]; Predict the class of a new message and interpret a misclassification [Applied]; Explain the bag-of-words limitation of discarding word order [Explained]

_Key takeaways:_ (1) CountVectorizer turns text into a matrix of token counts using a vocabulary learned from the training data. (2) The count representation is sparse and loses word order, a core limitation of the bag-of-words approach. (3) n-gram range, minimum document frequency, and maximum features control the size and content of the feature space. (4) Naive Bayes trains and predicts quickly compared to KNN and SVM on high-dimensional text features. (5) Strong training and test accuracy still allows individual predictions to be wrong on new messages.

**Taught how (8 patterns):** Comparison table contrasting methods; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Problem statement first (business framing); Real-world motivation with named domains; Small hand-computable toy dataset to teach mechanics; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 5: Evaluation Metrics

##### Evaluation Metrics
`unit_id: 03d78eba-b2d6-4d87-9cc4-4252d963ac11` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: c36fe709-4293-47ef-99a6-4507e97855e0`

**Teaches:** Evaluation Metrics

_Flow:_ 1) Motivation: spam vs not-spam filtering and the cost of each misclassification 2) The confusion matrix: TP, FP, FN, TN 3) Accuracy and when it is appropriate (balanced data) 4) Precision and when false positives are costly 5) Recall / sensitivity and when false negatives are costly 6) F1 score as the harmonic mean of precision and recall 7) Choosing which metric to prioritise from the business context (spam vs cancer)

_Outcomes (depth):_ Interpret a confusion matrix and its four cells [Explained]; Compute accuracy, precision, recall, and F1 from a confusion matrix [Applied]; Explain why accuracy can be misleading on imbalanced data [Explained]; Select the appropriate metric for a given business context (FP vs FN cost) [Deep-dive]

_Key takeaways:_ (1) The confusion matrix (TP, FP, FN, TN) is the basis for most classification metrics. (2) Accuracy suits balanced data; precision matters when false positives are costly; recall matters when false negatives are costly. (3) F1 score is the harmonic mean of precision and recall and is useful for imbalanced data. (4) Which metric to optimise depends on the business context and the relative cost of each error type.

**Taught how (10 patterns):** Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Build the concept by fixing the previous version's limitation; Definition callouts for terminology; Observe-and-interpret loop; Real-world motivation with named domains; Small hand-computable toy dataset to teach mechanics; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers; Worked example computed step by step

**Supporting content:** 1 reading

---

##### Evaluation Metrics Implementation
`unit_id: e9621c82-786f-4134-bea2-b919e244f567` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: c36fe709-4293-47ef-99a6-4507e97855e0`

**Teaches:** Evaluation Metrics, Train-Test Split

_Flow:_ 1) Set up the spam classification pipeline with vectorized features 2) Train a Multinomial Naive Bayes model 3) Predict classes for sample emails 4) Compute the confusion matrix 5) Calculate accuracy, precision, recall, and F1 score 6) Read the full classification report 7) Interpret each metric for the spam use case

_Outcomes (depth):_ Compute a confusion matrix for a classifier's predictions [Applied]; Calculate accuracy, precision, recall, and F1 score with scikit-learn [Applied]; Generate and read a per-class classification report [Applied]; Interpret the precision and recall trade-off for spam detection [Explained]; Train a Multinomial Naive Bayes model for text classification [Applied]

_Key takeaways:_ (1) The confusion matrix lays out true and false positives and negatives for a classifier. (2) Accuracy measures overall correct predictions but can mask errors on a minority class. (3) Precision is the share of predicted spam that is truly spam, while recall is the share of actual spam that is caught. (4) F1 score is the harmonic mean of precision and recall, balancing the two. (5) The classification report presents precision, recall, and F1 per class in a single view.

**Taught how (10 patterns):** Comparison table contrasting methods; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Problem statement first (business framing); Real-world motivation with named domains; Small hand-computable toy dataset to teach mechanics; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 6: Bias and Variance

##### Bias and Variance
`unit_id: a609d3f2-bd18-405a-a9ab-9bb2fdc1233c` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 4a796a5e-0edd-4831-a543-b610b9638b13`

**Teaches:** Bias and Variance

_Flow:_ 1) Define bias and variance as sources of model error 2) Relate high bias to underfitting and high variance to overfitting 3) Illustrate the bias-variance trade-off with error curves 4) List ways to reduce high bias 5) List ways to reduce high variance 6) Describe the characteristics of an ideal model 7) Introduce validation and its purpose 8) Compare hold-out, K-fold, and leave-one-out cross validation

_Outcomes (depth):_ Distinguish bias from variance and connect each to underfitting and overfitting [Explained]; Diagnose whether a model is underfitting or overfitting [Explained]; Choose strategies to reduce high bias or high variance [Explained]; Explain how validation estimates a model's generalization [Explained]; Compare hold-out, K-fold, and leave-one-out cross validation by cost and reliability [Deep-dive]

_Key takeaways:_ (1) Bias is error from an over-simplified model that underfits; variance is error from over-sensitivity to training data that overfits. (2) The bias-variance trade-off seeks the model complexity that minimizes validation error. (3) High bias is reduced with more complex models, more features, less regularization, or more training data. (4) High variance is reduced with validation, feature selection, regularization, or ensemble methods. (5) Validation estimates how well a model generalizes, and hold-out, K-fold, and leave-one-out differ in reliability and cost.

**Taught how (9 patterns):** Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Build the concept by fixing the previous version's limitation; Definition callouts for terminology; Forward reference to upcoming content; Paired theory-plus-implementation with live-demo cues; Real-world motivation with named domains; Recap bridge from prior session; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

#### Module 7: Decision Tree

##### Decision Tree Part 1
`unit_id: fd4cffb4-e8e2-44f9-99ef-601eebb976b8` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 9aeb3074-96fb-4d41-8b4f-46a31bc0f577`

**Teaches:** Decision Trees

_Flow:_ 1) Scenario: split 20 students into similar-interest groups (motivation) 2) Compare candidate splits (height / performance / classroom) by the purity of the resulting groups 3) Objective of a decision tree: reach pure nodes 4) Decision tree structure and terminology: root, decision/internal, leaf nodes, branches, decision rules 5) Measuring split quality: Entropy and Information Gain

_Outcomes (depth):_ Identify the parts of a decision tree (root, decision, leaf nodes) [Introduced]; Explain how a decision tree splits data to increase node purity [Explained]; Define entropy and information gain and their role in selecting splits [Explained]; Given candidate splits, determine which yields purer nodes [Deep-dive]

_Key takeaways:_ (1) A decision tree is a flowchart-like structure that splits data on feature values to reach purer nodes. (2) The best split produces the purest child nodes; purity is quantified with entropy. (3) Information gain is the reduction in entropy from a split and is used to choose the splitting feature. (4) Core terminology: root node, decision/internal nodes, branches, and leaf nodes (outcomes).

**Taught how (5 patterns):** Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Definition callouts for terminology; Small hand-computable toy dataset to teach mechanics; Worked example computed step by step

**Assessed by — Decision Tree — session: Decision Tree Part 1 (Phishing)** (EASY · accuracy > 85% · style 5/6)
- Tests: build a tree, output `submission_df`, beat 85%.
- Alignment: partial — meaningful threshold, but entropy/IG/split-choice untested.
- Value: decent. Changes: probe criterion/depth reasoning if feasible.

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Decision Tree Part 2
`unit_id: 6adab306-24cf-4efa-b695-74b3444697d9` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 9aeb3074-96fb-4d41-8b4f-46a31bc0f577`

**Teaches:** Decision Trees

_Flow:_ 1) Recap decision tree structure of root, decision nodes, and leaves 2) Define entropy and information gain 3) Present the steps of the ID3 algorithm 4) Break down the information gain and entropy formulas 5) Calculate dataset entropy on the Play Tennis data 6) Compute information gain per feature and pick the best split 7) Recursively build the tree from the highest-gain features 8) Contrast linear models with decision trees on non-linear data

_Outcomes (depth):_ Calculate the entropy of a dataset from its class proportions [Applied]; Compute the information gain of a candidate split feature [Applied]; Select the root split by comparing information gains across features [Applied]; Build a decision tree step by step with the ID3 algorithm [Applied]; Contrast decision trees with linear models on non-linear data [Deep-dive]

_Key takeaways:_ (1) Entropy measures the impurity or uncertainty of a set of labels, and a pure subset has entropy zero. (2) Information gain is the reduction in entropy achieved by splitting on a feature. (3) ID3 repeatedly chooses the feature with the highest information gain as the next split. (4) Splitting continues until every subset is pure or no features remain. (5) Decision trees can separate non-linear data that a single linear model cannot.

**Taught how (6 patterns):** Agenda-first, Key-Takeaways-last framing; Comparison table contrasting methods; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; Worked example computed step by step

**Supporting content:** 1 reading, 1 in_class_quiz

---

##### Decision Tree Implementation Part 1
`unit_id: 3ea7aa51-1f58-4ebc-a5c6-82c94ff06461` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 9aeb3074-96fb-4d41-8b4f-46a31bc0f577`

**Teaches:** Decision Trees

_Flow:_ 1) Define the telecom customer churn problem 2) Load the customer dataset 3) Explore the data with Sweet Viz 4) Split into train and test sets 5) Build and evaluate a decision tree model 6) Interpret results with a classification report and tree visualization

_Outcomes (depth):_ Frame customer churn as a binary classification problem [Explained]; Use Sweet Viz to explore a dataset [Introduced]; Evaluate a decision tree with a classification report [Introduced]; Visualize a trained decision tree [Introduced]

_Key takeaways:_ (1) The task is to predict whether a telecom customer will churn using demographic, service usage, and payment data. (2) Sweet Viz supports automated exploratory data analysis of the dataset. (3) A trained decision tree is assessed and interpreted through a classification report and a tree visualization.

**Taught how (4 patterns):** Agenda-first, Key-Takeaways-last framing; ML workflow skeleton in implementation notebooks; Problem statement first (business framing); Real-world motivation with named domains

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

##### Decision Tree Implementation Part 2
`unit_id: 6aef2838-d3d4-4742-9ebc-ab8f2fd649f2` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 9aeb3074-96fb-4d41-8b4f-46a31bc0f577`

**Teaches:** Decision Trees

_Flow:_ 1) Recap the churn problem and the role of validation 2) Review the steps of K-fold cross validation 3) Introduce stratified K-fold cross validation 4) Review the pros and cons of decision trees

_Outcomes (depth):_ Explain how K-fold cross validation rotates folds to estimate performance [Explained]; Explain how stratified K-fold preserves class balance across folds [Explained]; Identify when stratified sampling is preferable for imbalanced datasets [Explained]; Weigh the pros and cons of decision trees [Explained]

_Key takeaways:_ (1) K-fold cross validation trains and validates across K rotating folds to give a more reliable performance estimate. (2) Stratified K-fold keeps each fold's class proportions matching the full dataset, which helps with imbalanced data. (3) Decision trees are easy to interpret, handle both numerical and categorical data, need no feature scaling, and model non-linear relationships. (4) Decision trees are prone to overfitting, unstable to small data changes, and can be biased toward dominant classes.

**Taught how (6 patterns):** Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; Problem statement first (business framing); Real-world motivation with named domains; Recap bridge from prior session; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 in_class_quiz

---

#### Module 8: Hyper Parameter

##### HyperParameter Tuning Part1
`unit_id: e1c2f69c-dcd2-4830-b1ab-35ddc58c05d2` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 6cd04044-ed6e-424e-94ee-ecafb49ac68c`

**Teaches:** Class Imbalance Handling, Data Visualization, Evaluation Metrics, HyperParameter Tuning, Label Encoding, Missing Values & Outlier Treatment, Train-Test Split

_Flow:_ 1) Distinguish model parameters from hyperparameters 2) Survey ML and DL hyperparameters (learning rate, batch size, epochs, dropout, optimizer) 3) Motivate systematic tuning over manual trial-and-error 4) Compare Grid Search, Random Search, and Bayesian Optimization 5) Examine learning rate effects and decay schedules 6) Train a baseline decision tree and diagnose overfitting 7) Apply manual hyperparameter settings to constrain the tree 8) Tune the tree with GridSearchCV and RandomizedSearchCV 9) Compare accuracy across tuning methods

_Outcomes (depth):_ Distinguish model parameters from hyperparameters with examples [Explained]; Describe common deep learning hyperparameters and their effects [Explained]; Compare Grid Search, Random Search, and Bayesian Optimization [Deep-dive]; Explain how the learning rate and its schedules affect convergence [Explained]; Tune a decision tree using GridSearchCV and RandomizedSearchCV [Applied]; Diagnose overfitting from a train-test accuracy gap [Applied]

_Key takeaways:_ (1) Model parameters are learned from data during training, whereas hyperparameters are set before training and control how learning proceeds. (2) The learning rate sets the step size in gradient descent: too high causes divergence, too low converges slowly, and schedules can adapt it over epochs. (3) Grid Search exhaustively tests every combination, Random Search samples combinations for efficiency, and Bayesian Optimization uses past results to guide the next trial. (4) An unconstrained decision tree overfits, and tuning depth, split, and leaf limits narrows the gap between training and testing accuracy. (5) Tuning raised overall accuracy and minority-class precision, but a heavily imbalanced dataset still kept minority-class recall low.

**Taught how (13 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Paired theory-plus-implementation with live-demo cues; Recap bridge from prior session; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers

**Assessed by — Hyperparameter Tuning — session: HyperParameter Tuning Part 1** (MEDIUM · accuracy ≥ 65% (3-class) · style 6)
- Tests: output `submission_df`, beat 65%. No search artifact required.
- Alignment: **weak on the headline skill** — a default model passes; tuning (the whole point) is unverified.
- Value: low for intent. Changes: **highest-leverage fix** — require a `GridSearchCV`/`RandomizedSearchCV` object or best-params output, raise threshold.

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

##### HyperParameter Tuning Part2
`unit_id: c872eb3f-fedd-4806-8b22-52d7e5aace4a` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 6cd04044-ed6e-424e-94ee-ecafb49ac68c`

**Teaches:** Cross-Validation, Data Visualization, Evaluation Metrics, HyperParameter Tuning, Label Encoding, Missing Values & Outlier Treatment, Train-Test Split

_Flow:_ 1) Select model features by correlation with the target 2) Train a vanilla decision tree baseline 3) Validate stability with k-fold and stratified cross validation 4) Apply manual hyperparameter settings 5) Tune with GridSearchCV and RandomizedSearchCV 6) Compare train and test scores across methods 7) Compare two feature sets to judge feature-selection impact

_Outcomes (depth):_ Select model features based on their correlation with the target [Applied]; Evaluate a model with k-fold and stratified cross validation [Applied]; Tune a decision tree with grid and random search [Applied]; Compare model performance across two different feature sets [Deep-dive]; Interpret train versus test scores to assess generalization [Explained]

_Key takeaways:_ (1) Cross validation, including stratified folds, gives a more reliable performance estimate than a single train-test split. (2) Moving from a vanilla tree through cross validation and search tuning generally lifts test accuracy, though the gains are modest. (3) The choice of input features can matter as much as tuning: a stronger feature set raised test scores across every method. (4) Comparing models side by side across methods and feature sets clarifies which configuration generalizes best.

**Taught how (9 patterns):** Agenda-first, Key-Takeaways-last framing; Comparison table contrasting methods; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Problem statement first (business framing); Real-world motivation with named domains; Recap bridge from prior session; Teaching code carries inline explanatory comments

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz

---

#### Module 9: Capstone Project  (Loan Approval Prediction)

##### Problem Statement
`unit_id: 54f49af2-db7e-40e6-bcdc-76edf0e5f24a` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 20f23bd0-a447-4b2d-b2b6-be386c229fb2`

**Teaches:** _none tagged_

_Flow:_ 1) Frame the loan eligibility prediction problem 2) Recap the machine learning lifecycle stages 3) Define the classification objective 4) Generate hypotheses about factors affecting approval 5) Explore the dataset features and target variable

_Outcomes (depth):_ State the loan approval prediction task as a classification problem [Explained]; Recall the stages of the machine learning lifecycle [Introduced]; Generate hypotheses about factors influencing loan approval [Explained]; Describe each dataset feature and identify the target variable [Explained]

_Key takeaways:_ (1) The project goal is to automate loan eligibility by predicting approval, framed as a binary classification task. (2) Hypothesis generation lists plausible drivers of approval such as income, credit history, loan amount, and term before analyzing the data. (3) Understanding each feature's meaning and type upfront guides later preprocessing and modeling choices. (4) The target variable Loan_Status is categorical (approved or not), confirming a classification setup.

**Taught how (8 patterns):** Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; Hypothesis generation before analysis; Multi-session capstone project pipeline; Problem statement first (business framing); Real-world motivation with named domains; Recap bridge from prior session; Systematic feature-by-feature / column-by-column pass

**Assessed by — Capstone Project 1 — Classification (Malware)** (EASY · accuracy ≥ 0.80 · 7 test cases · style 1)
- Tests: runs, **a fitted sklearn classifier exists**, **≥1 EDA plot exists**, `submission_df` valid, ≥80%.
- Alignment: **good** — checks breadth (model + EDA + submission), resists shortcutting.
- Value: high. Changes: "EASY" label understates a full pipeline.

**Supporting content:** 1 reading, 1 code

---

##### EDA
`unit_id: 2c3a6ee0-d8a5-426f-890b-5021c6e45604` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 20f23bd0-a447-4b2d-b2b6-be386c229fb2`

**Teaches:** Data Visualization, Exploratory Data Analysis (EDA)

_Flow:_ 1) Load and inspect the loan dataset structure 2) Analyze the target variable distribution 3) Explore categorical and ordinal features univariately 4) Examine numerical feature distributions and outliers 5) Relate categorical features to the target with stacked bars 6) Bin numerical features and compare approval rates 7) Visualize feature correlations with a heatmap 8) Note class imbalance in the target

_Outcomes (depth):_ Load and profile a dataset's shape, types, and summary statistics [Applied]; Perform univariate analysis on categorical and numerical features [Applied]; Conduct bivariate analysis of features against the target [Applied]; Detect skewness and outliers from distribution and box plots [Applied]; Interpret a correlation heatmap among numerical features [Explained]; Recognize class imbalance in the target variable [Introduced]

_Key takeaways:_ (1) Univariate analysis summarizes single variables, showing most applicants are male, married, and have repaid prior debts. (2) Income and loan-amount features are right-skewed with visible outliers, hinting at transformations needed later. (3) Bivariate analysis with stacked proportion bars suggests good credit history and higher total income associate with higher approval rates. (4) Class imbalance in the target must be recognized because it can bias a model toward the majority class.

**Taught how (10 patterns):** Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Paired theory-plus-implementation with live-demo cues; Preprocessing justified by the algorithm's needs; Real-world motivation with named domains; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 code

---

##### Missing Values and Outliers Treatment
`unit_id: 05bc9d82-9bd3-4551-8ee6-3f60eaa01b98` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 20f23bd0-a447-4b2d-b2b6-be386c229fb2`

**Teaches:** Data Visualization, Missing Values & Outlier Treatment

_Flow:_ 1) Count missing values per feature 2) Impute categorical features with the mode 3) Impute Loan_Amount_Term using its most frequent value 4) Impute LoanAmount with the median to resist outliers 5) Apply the training imputations consistently to the test set 6) Identify outliers via boxplots, Z-score, and IQR 7) Reduce skew with a log transformation

_Outcomes (depth):_ Quantify missing values across a dataset [Applied]; Impute categorical and numerical missing values appropriately [Applied]; Justify median over mean imputation when outliers are present [Explained]; Identify outliers using Z-score, IQR, and boxplots [Explained]; Apply a log transformation to reduce skewness [Applied]

_Key takeaways:_ (1) Categorical missing values are typically filled with the mode, while numerical ones use mean or median depending on the distribution. (2) The median is preferred over the mean for imputing loan amount because outliers distort the mean. (3) Test-set values are imputed using statistics computed from the training set to avoid leaking information. (4) Outliers can be detected with Z-score, IQR, or plots, and treated by removal, capping, or transformation. (5) A log transformation compresses large values and pulls a right-skewed distribution closer to normal.

**Taught how (8 patterns):** Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Preprocessing justified by the algorithm's needs; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 code

---

##### Feature Engineering - 1
`unit_id: 65f3e71b-4e79-4a30-8872-1800ef418038` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 20f23bd0-a447-4b2d-b2b6-be386c229fb2`

**Teaches:** Data Visualization, Feature Engineering, Label Encoding, Missing Values & Outlier Treatment, One-Hot Encoding

_Flow:_ 1) Set the loan ID as the DataFrame index 2) Drop helper columns created during analysis 3) Binary-encode two-value categorical columns 4) Convert the Dependents category to numeric 5) One-hot encode a multi-category nominal feature 6) Label encode a categorical feature 7) Contrast label encoding with one-hot encoding

_Outcomes (depth):_ Binary-encode two-valued categorical columns [Applied]; One-hot encode a nominal categorical feature [Applied]; Apply label encoding to a categorical column [Applied]; Choose between label and one-hot encoding by data type [Deep-dive]; Prepare a numeric feature matrix consistent across train and test [Applied]

_Key takeaways:_ (1) Categorical features must be numerically encoded before most machine learning algorithms can use them. (2) Binary encoding maps two-value columns to 0 and 1, a compact choice for yes/no style features. (3) Label encoding assigns integers and implies an order, so it suits ordinal data but can mislead on nominal data. (4) One-hot encoding creates a binary column per category, avoiding a false ordering at the cost of more columns.

**Taught how (8 patterns):** Agenda-first, Key-Takeaways-last framing; Comparison table contrasting methods; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Small hand-computable toy dataset to teach mechanics; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 code

---

##### Evaluation Metrics
`unit_id: 4b26cfde-475d-4e35-a781-fb2927608f66` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 20f23bd0-a447-4b2d-b2b6-be386c229fb2`

**Teaches:** Evaluation Metrics

_Flow:_ 1) Read a confusion matrix's four outcomes 2) Compute accuracy and note its limitation on imbalanced data 3) Define precision and when false positives matter 4) Define recall and when false negatives matter 5) Combine precision and recall into the F1 score 6) Choose a primary metric for the problem context 7) Recap candidate ML models for the project

_Outcomes (depth):_ Interpret the four cells of a confusion matrix [Explained]; Compute accuracy, precision, recall, and F1 score [Explained]; Select precision or recall as the priority metric for a scenario [Deep-dive]; Explain why F1 suits imbalanced classification [Explained]

_Key takeaways:_ (1) The confusion matrix breaks predictions into true and false positives and negatives, the basis for the other metrics. (2) Accuracy alone can mislead on imbalanced data, so precision, recall, and F1 give a fuller picture. (3) Precision matters when false positives are costly, while recall matters when missing true positives is costly. (4) The F1 score balances precision and recall through their harmonic mean, making it useful under class imbalance.

**Taught how (5 patterns):** Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; Real-world motivation with named domains; Recap bridge from prior session; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Build Base ML model
`unit_id: 8782b223-5efb-4e91-b62a-90843a3402f3` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 20f23bd0-a447-4b2d-b2b6-be386c229fb2`

**Teaches:** Data Visualization, Evaluation Metrics, Label Encoding, Missing Values & Outlier Treatment, One-Hot Encoding, Train-Test Split

_Flow:_ 1) Split features and target from the encoded data 2) Create train and test partitions 3) Build and tune a KNN classifier over k values 4) Build a linear SVM classifier 5) Build a shallow decision tree and visualize it 6) Generate predictions on the held-out test set 7) Compare baseline model accuracies

_Outcomes (depth):_ Split prepared data into training and test sets [Applied]; Train a KNN classifier and select k from accuracy trends [Applied]; Train linear SVM and decision tree baseline classifiers [Applied]; Generate and save predictions on unseen data [Applied]; Compare baseline models by accuracy [Applied]

_Key takeaways:_ (1) A base model establishes a performance benchmark before any optimization is attempted. (2) KNN accuracy varies with the number of neighbors, so scanning k values and plotting accuracy helps pick a stable choice. (3) Different algorithms such as KNN, SVM, and decision trees can be trained on the same prepared data and compared directly. (4) A bar chart of accuracies makes it easy to see which baseline model performs best on the loan data.

**Taught how (6 patterns):** Agenda-first, Key-Takeaways-last framing; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments

**Supporting content:** 1 reading, 1 code

---

##### Feature Engineering - 2
`unit_id: 3251fa34-213a-4b8b-83d9-c8cfe19a7315` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 20f23bd0-a447-4b2d-b2b6-be386c229fb2`

**Teaches:** Data Visualization, Evaluation Metrics, Feature Engineering, Label Encoding, Missing Values & Outlier Treatment, One-Hot Encoding, Train-Test Split

_Flow:_ 1) Log-transform skewed income and loan-amount features 2) Apply a square-root transform to coapplicant income 3) Drop the original skewed columns and mirror on test 4) Establish decision tree and SVM baselines 5) Rank features with Leave-One-Feature-Out importance 6) Interpret which features drive model performance 7) Derive actionable feature-selection insights

_Outcomes (depth):_ Apply log and square-root transforms to skewed features [Applied]; Compute feature importance using Leave-One-Feature-Out [Applied]; Rank features by their performance-drop contribution [Applied]; Interpret importance scores to guide feature selection [Explained]; Weigh the trade-offs of the LOFO approach [Deep-dive]

_Key takeaways:_ (1) Log and square-root transforms reduce skew in income and loan-amount features before importance analysis. (2) Feature importance reveals each feature's contribution, aiding interpretability and feature selection. (3) Leave-One-Feature-Out removes one feature at a time and measures the accuracy drop, where a larger drop means greater importance. (4) Credit history dominated importance while most other features contributed little to accuracy. (5) LOFO is model-agnostic and direct but computationally expensive and blind to feature interactions.

**Taught how (9 patterns):** Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Definition callouts for terminology; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code

---

##### Build ML model and Conclusion
`unit_id: 1e477106-0cb0-4cca-adb5-39295906d2d2` · `course_id: 919bb576-966a-48df-a1e0-b8b071fc7f69` · `topic_id: 20f23bd0-a447-4b2d-b2b6-be386c229fb2`

**Teaches:** Data Visualization, Evaluation Metrics, HyperParameter Tuning, Label Encoding, Missing Values & Outlier Treatment, One-Hot Encoding, Train-Test Split

_Flow:_ 1) Select the most informative features for the final model 2) Split the reduced data into train and test sets 3) Tune a decision tree with GridSearchCV 4) Tune the decision tree with RandomizedSearchCV 5) Tune an SVM over C, kernel, and gamma 6) Evaluate tuned models with accuracy and classification reports

_Outcomes (depth):_ Select a final feature subset informed by importance analysis [Applied]; Tune a decision tree with grid and randomized search [Applied]; Tune an SVM across C, kernel, and gamma [Applied]; Evaluate tuned models with accuracy and classification reports [Applied]; Complete an end-to-end classification workflow for loan approval [Applied]

_Key takeaways:_ (1) Building on the earlier importance analysis, the final model uses a focused subset of the most informative features. (2) GridSearchCV searches a fixed parameter grid while RandomizedSearchCV samples from parameter distributions. (3) SVM tuning spans the regularization strength, the kernel choice, and the gamma coefficient. (4) Cross-validated search selects the best hyperparameters, and the held-out test set gives the final performance estimate.

**Taught how (9 patterns):** Diagnose-then-remedy pairing; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments

**Supporting content:** 1 reading, 1 code

---

### Supervised Learning: Regression

#### Module 10: Simple Linear Regression

##### Simple Linear Regression
`unit_id: b3e07b06-c67d-424d-bb81-1890ed49b5fc` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: d69961f7-6423-4be1-97b2-e497ce48206b`

**Teaches:** Simple Linear Regression

_Flow:_ 1) Recap regression as modeling continuous data 2) Define simple linear regression and the line of best fit 3) Introduce the y = mx + c equation and each of its terms 4) List the types of linear regression and real-world applications 5) Explain Ordinary Least Squares and the Sum of Squared Errors 6) Walk through the OLS steps: means, centroid, slope, intercept 7) Compute slope and intercept on a worked bill-and-tip example 8) Compare the cost functions MAE, MSE, and RMSE

_Outcomes (depth):_ Define simple linear regression and identify each term in the y = mx + c equation [Explained]; Compute the slope and intercept of a best-fit line using the OLS formulas [Applied]; Explain how Ordinary Least Squares minimizes squared errors to fit the line [Explained]; Calculate MAE, MSE, and RMSE for a set of predictions [Applied]; Distinguish MAE, MSE, and RMSE by how they treat errors and units [Explained]

_Key takeaways:_ (1) Simple linear regression models a straight-line relationship between one predictor and a continuous outcome. (2) The best-fit line is the one that minimizes the sum of squared errors between actual and predicted values. (3) OLS finds the slope and intercept from deviations around the means of X and Y, and the fitted line passes through the centroid. (4) MAE, MSE, and RMSE quantify prediction error, with MSE penalizing larger errors more heavily than MAE. (5) RMSE expresses error in the same units as the target, making it easier to interpret.

**Taught how (12 patterns):** Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Metric interpreted in plain-language real-world units; Paired theory-plus-implementation with live-demo cues; Real-world motivation with named domains; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; Worked example computed step by step

**Assessed by — Simple Linear Regression** (MEDIUM · RMSE ≤ 22 · 3 test cases · style 1)
- Tests: **`LinearRegression` present** + RMSE ≤ 22 (performance-weighted 50%).
- Alignment: **good** — verifies the estimator type. Value: good. Changes: none urgent.

**Supporting content:** 2 in_class_quiz, 2 mcq, 1 reading

---

##### SLR Implementation
`unit_id: e81ca31c-2b68-413e-b832-cb418922f7f8` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: d69961f7-6423-4be1-97b2-e497ce48206b`

**Teaches:** Assumptions of Linear Regression, Data Visualization, Regression Metrics (R2/MSE/RMSE/MAE), Simple Linear Regression, Train-Test Split

_Flow:_ 1) Define the sales-prediction problem and the modeling workflow 2) Load and inspect the advertising dataset 3) Check for missing values and outliers using boxplots 4) Explore feature-target relationships with pairplots and a correlation heatmap 5) Split the data into training and test sets 6) Fit a simple linear regression by adding a constant and applying OLS 7) Evaluate residuals for normality and patterns 8) Predict on the test set and measure error with RMSE, then review pros and cons

_Outcomes (depth):_ Load, inspect, and clean a dataset before modeling [Applied]; Visualize feature-target relationships using pairplots and a correlation heatmap [Applied]; Fit a simple linear regression model with ordinary least squares [Applied]; Evaluate a fitted model by examining residual distribution and test-set error [Applied]; Summarize the pros and cons of simple linear regression [Explained]

_Key takeaways:_ (1) A regression workflow moves from problem definition through data inspection, EDA, model fitting, and evaluation. (2) Adding a constant term is required to fit an intercept when using ordinary least squares. (3) The fitted model expresses the target as an intercept plus a slope times the predictor. (4) Residuals should center near zero and follow a roughly normal distribution for the fit to be trustworthy. (5) Simple linear regression is easy to interpret and a strong baseline, but it uses only one predictor and is sensitive to outliers.

**Taught how (8 patterns):** Agenda-first, Key-Takeaways-last framing; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Paired theory-plus-implementation with live-demo cues; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code

---

#### Module 11: Multiple Linear Regression

##### Multiple Linear Regression
`unit_id: a93ab3de-76c1-4241-80d6-12b455feae96` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: ff741a2e-89a5-4e6c-aba8-1fa636260e33`

**Teaches:** Data Visualization, Multiple Linear Regression, One-Hot Encoding, Regression Metrics (R2/MSE/RMSE/MAE), Train-Test Split

_Flow:_ 1) Recap simple linear regression 2) Define multiple linear regression as an extension with several predictors 3) Break down the MLR equation and its coefficients 4) Survey applications across business, healthcare, and finance 5) Work through a house-price example with multiple features 6) Encode categorical variables with one-hot encoding 7) Split the data and fit a multiple regression model 8) Evaluate with MSE and apply a log transform to improve the fit

_Outcomes (depth):_ Write the multiple linear regression equation and interpret its coefficients [Explained]; Predict an outcome from several features using fitted coefficients [Applied]; Encode categorical variables for use in a regression model [Applied]; Fit and evaluate a multiple linear regression model on a dataset [Applied]; Apply a log transformation to a skewed target and reverse it for predictions [Applied]

_Key takeaways:_ (1) Multiple linear regression models one outcome as a linear combination of two or more predictors. (2) Each coefficient represents the change in the outcome per unit change in its predictor, holding the others fixed. (3) Categorical predictors must be converted to numeric form via one-hot encoding before fitting. (4) A large error metric can signal the need for feature engineering, scaling, or transforming the target. (5) Applying a log transform to a skewed target can reduce error, with predictions transformed back to the original scale.

**Taught how (14 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Definition callouts for terminology; Diagnose-then-remedy pairing; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Problem statement first (business framing); Real-world motivation with named domains; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments; Term-by-term decomposition of a formula; Worked example computed step by step

**Assessed by — Multiple Linear Regression** (EASY · R² ≥ 0.90 · style 5)
- Tests: `submission_df`, R² ≥ 0.90 (high bar forces correct encoding/preprocessing).
- Alignment: reasonable. Value: good. Changes: consider verifying a linear model.

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 12: Polynomial Regression

##### Polynomial Regression
`unit_id: aff13bca-f90f-4af1-a9bb-423affbe022a` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: de6ddf16-975d-42f4-ad2a-a8814478a824`

**Teaches:** Polynomial Regression

_Flow:_ 1) Recap multiple linear regression 2) Define polynomial regression for non-linear relationships 3) Present the polynomial equation and its higher-power terms 4) Motivate polynomial regression through flexibility and curvature 5) Contrast underfitting and overfitting and the bias-variance balance 6) Introduce regularization and hyperparameter tuning to choose the degree 7) Work through a quadratic plant-growth example

_Outcomes (depth):_ Define polynomial regression and write its general equation [Explained]; Explain why higher-power terms let the model capture non-linear patterns [Explained]; Distinguish underfitting from overfitting in terms of bias and variance [Explained]; Predict an outcome using a fitted quadratic model [Applied]; Identify regularization and degree tuning as ways to control overfitting [Introduced]

_Key takeaways:_ (1) Polynomial regression fits a curve by including higher powers of the predictor, capturing non-linear relationships. (2) The model stays linear in its coefficients even though the relationship with X is non-linear. (3) Choosing the polynomial degree means balancing underfitting, which is too simple, against overfitting, which fits noise. (4) Regularization techniques such as Ridge and Lasso help control overfitting. (5) The appropriate degree is selected through hyperparameter tuning by minimizing validation error.

**Taught how (11 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Definition callouts for terminology; Diagnose-then-remedy pairing; Real-world motivation with named domains; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers; Worked example computed step by step

**Assessed by — Polynomial Regression (Quality Rating)** (MEDIUM · MSE < 17 **and** R² > 0.90 · style 5)
- Tests: dual threshold (MSE and R²) — harder to game, rewards the right degree.
- Alignment: good. Value: high. Changes: none urgent.

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 13: Gradient Descent

##### Gradient Descent Part - 1
`unit_id: 6e95dd9b-3976-46b1-8994-65b7efa85ff4` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 352f8931-da2f-4a21-a654-9792710297e0`

**Teaches:** Gradient Descent

_Flow:_ 1) Build intuition for gradient descent with everyday analogies 2) Contrast Ordinary Least Squares with gradient descent 3) Define gradient descent as iterative cost minimization 4) Outline the gradient descent process step by step 5) Define the MSE cost function for a linear model 6) Explain why the one-half factor simplifies the gradient 7) Derive the partial derivatives for slope and intercept 8) State the parameter update rules with a learning rate 9) Describe convergence and stopping criteria

_Outcomes (depth):_ Explain how gradient descent minimizes a cost function iteratively [Explained]; Compare gradient descent with OLS on speed, exactness, and scalability [Deep-dive]; Derive the partial derivatives of the MSE cost with respect to slope and intercept [Deep-dive]; State and interpret the parameter update rules that use a learning rate [Explained]; Identify the stopping criteria that signal convergence [Explained]

_Key takeaways:_ (1) Gradient descent iteratively adjusts parameters in the direction of steepest descent to minimize a cost function. (2) Unlike OLS, which computes an exact solution directly, gradient descent is approximate but scales to large datasets. (3) The learning rate controls the step size of each parameter update. (4) Scaling the MSE cost by one-half cancels the factor of 2 from differentiation, simplifying the gradients. (5) The algorithm stops when the cost change falls below a threshold, the gradient nears zero, or a maximum iteration count is reached.

**Taught how (8 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Comparison table contrasting methods; Definition callouts for terminology; Flag optional or simplified mathematical depth; Term-by-term decomposition of a formula; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Gradient Descent Part - 2
`unit_id: 47687052-3500-49c9-877f-08180848bad0` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 352f8931-da2f-4a21-a654-9792710297e0`

**Teaches:** Gradient Descent

_Flow:_ 1) Introduce the three variants of gradient descent 2) Explain batch gradient descent and its trade-offs 3) Explain stochastic gradient descent and its trade-offs 4) Explain mini-batch gradient descent and its trade-offs 5) Compare the three variants on speed, stability, and use case 6) Define R-squared as the proportion of variance explained 7) Define adjusted R-squared and why it corrects for the number of predictors

_Outcomes (depth):_ Distinguish batch, stochastic, and mini-batch gradient descent by their data usage [Explained]; Compare the three variants on speed, stability, and best use case [Deep-dive]; Select an appropriate gradient descent variant for a given dataset size [Explained]; Interpret R-squared as the proportion of variance explained [Explained]; Explain why adjusted R-squared corrects for the number of predictors [Explained]

_Key takeaways:_ (1) Gradient descent variants differ by how much data they use per update: the full dataset, one point, or a mini-batch. (2) Batch gradient descent is stable but slow, stochastic is fast but noisy, and mini-batch balances the two. (3) Mini-batch gradient descent suits most practical scenarios and takes advantage of vectorized hardware. (4) R-squared measures the proportion of variance in the target explained by the model, ranging from 0 to 1. (5) Adjusted R-squared penalizes adding predictors, preventing an overestimation of the fit.

**Taught how (8 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Metric interpreted in plain-language real-world units; Term-by-term decomposition of a formula; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Gradient Descent
`unit_id: ab850cc2-fa88-4701-8823-df24750d1f2e` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 352f8931-da2f-4a21-a654-9792710297e0`

**Teaches:** Data Visualization, Feature Scaling, Gradient Descent, One-Hot Encoding, Regression Metrics (R2/MSE/RMSE/MAE), Standardization (Z-score Scaling), Train-Test Split

_Flow:_ 1) Define the car-price prediction problem 2) Prepare features by dropping a high-cardinality column and deriving car age 3) Explore the data and detect outliers with plots 4) Encode categorical variables and analyze correlations 5) Split the data and scale features on the training set only 6) Train a linear regression model and evaluate it 7) Implement gradient descent from scratch with weight updates 8) Track and plot the cost function across epochs 9) Interpret convergence and the effect of the learning rate

_Outcomes (depth):_ Prepare and encode features for a regression dataset [Applied]; Scale features correctly by fitting the scaler on training data only [Applied]; Implement gradient descent from scratch to update model weights [Applied]; Plot and interpret the cost function over training epochs [Applied]; Judge whether a chosen learning rate produces stable convergence [Applied]

_Key takeaways:_ (1) Feature preparation includes dropping high-cardinality columns and deriving more useful attributes, such as age from the model year. (2) Scaling must be fit on the training set only to avoid leaking test information into the model. (3) Gradient descent can be implemented from scratch by initializing weights, computing gradients, and updating iteratively. (4) A bias column of ones is appended to the feature matrix so the model can learn an intercept. (5) A cost curve that steadily decreases and then flattens indicates the algorithm is converging.

**Taught how (8 patterns):** Diagnose-then-remedy pairing; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Assessed by — Gradient Descent** (MEDIUM · RMSE ≤ 5.5 · 4 test cases · style 1)
- Tests: **`isinstance(model, SGDRegressor)`** + all 4 features scaled (StandardScaler used) + intercept in range + RMSE ≤ 5.5.
- Alignment: **strong** — verifies estimator, scaling, and convergence. The best-aligned mechanics assignment.
- Value: high. Changes: none urgent.

**Supporting content:** 1 reading, 1 code, 1 mcq

---

#### Module 14: Assumptions of Linear Regression

##### Assumptions of Linear Regression
`unit_id: 270fe1ce-03c5-4cf2-bac2-ad419a873157` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 4556f92c-fc01-4401-b00a-0a5467ee3bc3`

**Teaches:** Assumptions of Linear Regression

_Flow:_ 1) Frame the assumptions as the rules that make regression valid 2) Explain linearity and how to detect its violation 3) Explain independence of observations and residuals 4) Explain homoscedasticity as constant residual variance 5) Explain the requirement of no multicollinearity among predictors 6) Define the Variance Inflation Factor and its formula 7) Work through a VIF example to spot a problematic predictor 8) Explain normality of residuals via histograms and Q-Q plots

_Outcomes (depth):_ List the five assumptions of linear regression [Explained]; Detect non-linearity and non-constant variance using residual plots [Explained]; Compute and interpret the Variance Inflation Factor to detect multicollinearity [Applied]; Check normality of residuals using histograms and Q-Q plots [Explained]; Explain the consequence of violating each assumption [Explained]

_Key takeaways:_ (1) Linear regression relies on five assumptions: linearity, independence, homoscedasticity, no multicollinearity, and normal residuals. (2) Residual plots reveal non-linearity and non-constant variance when patterns appear. (3) Homoscedasticity requires the spread of residuals to stay constant across levels of the predictors. (4) Multicollinearity is detected with the Variance Inflation Factor, where values above 10 signal a problem. (5) Residuals should be normally distributed, which can be checked with histograms and Q-Q plots.

**Taught how (7 patterns):** Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Definition callouts for terminology; Diagnose-then-remedy pairing; Term-by-term decomposition of a formula; Valid-case-versus-violation contrast; Worked example computed step by step

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Assumptions of Linear Regression Implementation
`unit_id: edc853f3-d899-4433-b43d-917a9f4e4868` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 4556f92c-fc01-4401-b00a-0a5467ee3bc3`

**Teaches:** Assumptions of Linear Regression, Data Visualization, Feature Scaling, Multicollinearity (VIF), One-Hot Encoding, Regression Metrics (R2/MSE/RMSE/MAE), Standardization (Z-score Scaling), Train-Test Split

_Flow:_ 1) Prepare the car dataset and train a linear regression model 2) Check linearity with actual-vs-predicted and residual plots 3) Check normality of residuals with a histogram and Q-Q plot 4) Compute VIF to check for multicollinearity 5) Check homoscedasticity with a residuals-vs-predicted plot 6) Interpret each diagnostic and propose remedies

_Outcomes (depth):_ Generate residual and actual-vs-predicted plots to test linearity [Applied]; Assess residual normality using a histogram and Q-Q plot [Applied]; Compute per-feature VIF to diagnose multicollinearity [Applied]; Test homoscedasticity with a residuals-vs-predicted plot [Applied]; Recommend remedies such as transformations when assumptions fail [Applied]

_Key takeaways:_ (1) Regression assumptions can be validated in code using diagnostic plots and metrics on a fitted model. (2) Actual-vs-predicted and residual plots reveal whether the linearity assumption holds. (3) A histogram together with a Q-Q plot assesses whether residuals are normally distributed. (4) The Variance Inflation Factor quantifies multicollinearity per feature against interpretation thresholds. (5) A curved trend line in a residuals-vs-predicted plot signals a violation of homoscedasticity.

**Taught how (9 patterns):** Diagnose-then-remedy pairing; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments; Valid-case-versus-violation contrast; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code, 1 mcq

---

#### Module 15: Regularization

##### Regularization
`unit_id: 3473b49f-8226-462e-9ecf-1c5fad041005` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: ec046aac-2c54-4607-a9aa-8f5eb62105f0`

**Teaches:** Regularization (L1/L2)

_Flow:_ 1) Define regularization as a technique to prevent overfitting 2) Explain adding a penalty term to the cost function 3) Introduce L2 (Ridge) with a squared-coefficient penalty 4) Show the Ridge loss function and the role of lambda 5) Introduce L1 (Lasso) with an absolute-value coefficient penalty 6) Show the Lasso loss function and its feature-selection effect 7) Explain how Lasso produces sparse models by zeroing coefficients 8) Compare Lasso and Ridge on coefficients, use cases, and interpretability

_Outcomes (depth):_ Explain how a regularization penalty discourages model complexity to reduce overfitting [Explained]; Distinguish L1 and L2 regularization by their penalty terms and effect on coefficients [Explained]; Describe how the lambda hyperparameter controls regularization strength [Explained]; Identify how Lasso produces sparse models by setting coefficients to zero [Explained]; Select Lasso versus Ridge based on feature-selection and interpretability needs [Deep-dive]

_Key takeaways:_ (1) Regularization adds a penalty on coefficient magnitude to the loss function to discourage overly complex models and reduce overfitting. (2) The lambda hyperparameter controls regularization strength, with larger values shrinking coefficients more aggressively. (3) Ridge (L2) penalizes squared coefficients, shrinking them toward zero but never exactly to zero. (4) Lasso (L1) penalizes absolute coefficients, driving some exactly to zero and thereby performing automatic feature selection. (5) Lasso yields sparse, more interpretable models suited to few-important-feature cases, while Ridge suits many correlated features.

**Taught how (6 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Term-by-term decomposition of a formula; When-to-use guidance with concrete triggers

**Assessed by — Regularisation (Life Expectancy)** (MEDIUM · Ridge RMSE ≤ 5.0, Lasso RMSE ≤ 6.0 · style 1)
- Tests: **Ridge model present + Lasso model present** + separate RMSE bars for each.
- Alignment: **excellent** — directly tests both L1 and L2, the exact session content.
- Value: high. Changes: none urgent.

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Regularization Implementation
`unit_id: 65717cd7-b1e0-4eb1-89e1-5e737e3f6b5a` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: ec046aac-2c54-4607-a9aa-8f5eb62105f0`

**Teaches:** Assumptions of Linear Regression, Data Visualization, Feature Scaling, Multicollinearity (VIF), One-Hot Encoding, Regression Metrics (R2/MSE/RMSE/MAE), Regularization (L1/L2), Standardization (Z-score Scaling), Train-Test Split

_Flow:_ 1) Load the car-price dataset and prepare features (drop high-cardinality name, convert year to age) 2) Handle missing values and inspect duplicates 3) Explore data with univariate, bivariate, and outlier plots 4) Dummy-encode categorical variables and inspect correlations for multicollinearity 5) Split into train and test, then scale features fitting the scaler on training data only 6) Train and evaluate a baseline linear regression with MAE, MSE, RMSE, and R2 7) Check linear regression assumptions (linearity, residual normality, VIF, homoscedasticity) 8) Implement gradient descent from scratch and track cost convergence 9) Fit Ridge and Lasso, compare train and test error, and inspect Lasso feature selection

_Outcomes (depth):_ Build an end-to-end regression pipeline from raw car-price data to a trained model [Applied]; Engineer features by dropping high-cardinality columns and converting year to age [Applied]; Prevent data leakage by fitting the scaler on the training set only [Applied]; Diagnose linear regression assumptions using residual plots, Q-Q plots, and VIF [Applied]; Implement gradient descent from scratch and interpret its cost-convergence curve [Applied]; Compare Ridge and Lasso against a baseline model and inspect Lasso feature selection [Applied]

_Key takeaways:_ (1) A full regression pipeline requires feature preparation, encoding, scaling, and leakage-free train/test handling before modeling. (2) Fitting the StandardScaler only on the training set prevents test-set information from leaking into the model. (3) Linear regression validity rests on assumptions (linearity, normal residuals, low multicollinearity, homoscedasticity) verified with diagnostic plots and VIF. (4) Ridge and Lasso can be fit with a chosen alpha and compared against a baseline linear model using train and test MSE and RMSE. (5) Lasso drives some coefficients to zero, so inspecting retained coefficients reveals which predictors it keeps.

**Taught how (8 patterns):** ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 16: Logistic Regression

##### Logistic Regression
`unit_id: 2c658051-df41-40b8-a20c-bb1e5574ea9a` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 43eda1e9-b1e4-4224-96b2-dd0e95fc21f1`

**Teaches:** Logistic Regression

_Flow:_ 1) Introduce logistic regression for binary classification via probability 2) Motivate with a diabetic versus non-diabetic example 3) Show why linear regression fails for bounded probabilities 4) Introduce the sigmoid function mapping z to a 0-1 probability 5) Explain odds and the log-odds (logit) transformation 6) Relate probability, odds, and log-odds (S-curve, exponential, linear) 7) Contrast linear and logistic regression on output and cost function 8) Introduce the log-loss (cross-entropy) cost function and its behavior 9) Survey real-world applications of logistic regression

_Outcomes (depth):_ Explain how logistic regression predicts the probability of a binary class [Explained]; Justify why linear regression is inappropriate for classification tasks [Explained]; Describe how the sigmoid function bounds predictions between 0 and 1 [Explained]; Convert between probability, odds, and log-odds for a given outcome [Applied]; Interpret the log-loss cost function and how it penalizes predictions when the true label is 1 versus 0 [Explained]; Contrast linear and logistic regression by output and cost function [Deep-dive]

_Key takeaways:_ (1) Logistic regression models the probability of a binary outcome by passing a linear combination of features through the sigmoid function. (2) Linear regression is unsuitable for classification because its unbounded output can fall outside the valid 0-1 probability range. (3) The log-odds (logit) transformation linearizes the relationship between features and the target and ranges from negative to positive infinity. (4) The sigmoid function converts log-odds back into interpretable probabilities bounded between 0 and 1. (5) Logistic regression is trained with log loss (cross-entropy) rather than mean squared error, penalizing confident wrong predictions heavily.

**Taught how (10 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Paired theory-plus-implementation with live-demo cues; Real-world motivation with named domains; Term-by-term decomposition of a formula; Worked example computed step by step

**Assessed by — Logistic Regression (Heart Stroke)** (EASY · accuracy ≥ 65% · style 6)
- Tests: `submission_df`, beat 65%. No technique verified.
- Alignment: **weak** — lenient, technique-agnostic. Value: workflow only. Changes: verify a LogisticRegression, raise threshold.

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Logistic Regression Implementation
`unit_id: e76ab4a2-54ab-4553-81b1-b48d09253634` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 43eda1e9-b1e4-4224-96b2-dd0e95fc21f1`

**Teaches:** Data Visualization, Logistic Regression, Missing Values & Outlier Treatment, Train-Test Split

_Flow:_ 1) Define the diabetes classification problem and load the dataset 2) Explore target balance and feature distributions 3) Run bivariate analysis of features against the outcome 4) Treat physiologically impossible zeros as missing and impute by median or mean per distribution 5) Examine feature correlations 6) Split the data and train a logistic regression classifier 7) Inspect learned coefficients, intercept, and predicted probabilities 8) Evaluate with log loss and interpret the result 9) Review the pros and cons of logistic regression

_Outcomes (depth):_ Build and train a logistic regression classifier on a medical dataset [Applied]; Detect and impute physiologically impossible zero values using median or mean by distribution [Applied]; Extract coefficients, intercept, and class probabilities from a fitted model [Applied]; Evaluate a classifier with log loss and interpret a high value [Applied]; Summarize the pros and cons of logistic regression [Explained]

_Key takeaways:_ (1) Building a logistic regression classifier follows a pipeline of problem definition, EDA, cleaning, training, and evaluation. (2) Impossible zero values can be treated as missing and imputed with the median for skewed features and the mean for near-normal features. (3) A trained logistic regression exposes coefficients, an intercept, and class probabilities that support interpretation. (4) Log loss quantifies classification quality, where a high value signals poorly calibrated or inaccurate probability predictions. (5) Logistic regression is simple and interpretable but assumes linearity in the log-odds and is sensitive to unscaled and imbalanced data.

**Taught how (8 patterns):** Agenda-first, Key-Takeaways-last framing; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 17: KNN Regression

##### KNN Regression
`unit_id: e77e2a55-eb64-4f0d-a030-2c4ea40f121b` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: e7d63313-b33e-4738-ab88-df3429bad9f5`

**Teaches:** Data Visualization, HyperParameter Tuning, KNN Regression, Regression Metrics (R2/MSE/RMSE/MAE), Train-Test Split

_Flow:_ 1) Recap KNN and introduce KNN regression for continuous targets 2) Explain predicting a value by averaging the k nearest neighbors' targets 3) Walk through the model-building steps (choose k, compute distances, aggregate, predict) 4) Load and inspect the advertising sales dataset 5) Explore features with outlier boxplots, pairplots, and correlation 6) Train a KNeighborsRegressor and evaluate with MSE 7) Sweep k to compare training and testing error and select a value 8) Tune neighbors, weighting, and distance metric with GridSearchCV 9) Review applications, pros, and cons

_Outcomes (depth):_ Explain how KNN regression predicts a continuous value from neighboring points [Explained]; Outline the steps of the KNN regression algorithm [Explained]; Train and evaluate a KNeighborsRegressor on a sales dataset [Applied]; Compare training and testing error across values of k to select k [Applied]; Tune KNN hyperparameters with GridSearchCV using cross-validation [Applied]; Assess the pros and cons of KNN regression [Explained]

_Key takeaways:_ (1) KNN regression is an instance-based, non-parametric method that predicts a continuous value from the average of its k nearest neighbors' targets. (2) Prediction proceeds by choosing k, computing distances, aggregating neighbor targets, and assigning the result. (3) The choice of k trades off underfitting and overfitting, which can be studied by comparing training and testing error across k values. (4) Grid search over number of neighbors, weighting scheme, and distance metric tunes a KNN regressor using cross-validation. (5) KNN is simple and flexible but computationally costly on large data and sensitive to k, distance metric, noise, and irrelevant features.

**Taught how (10 patterns):** Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Paired theory-plus-implementation with live-demo cues; Problem statement first (business framing); Real-world motivation with named domains; Recap bridge from prior session; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Assessed by — KNN, SVM, DT — session: KNN Regression** (MEDIUM · KNN R²>65%, DT R²>60%, SVM MSE>1 · style 2)
- Tests: 3-column submission (KNN/SVM/Decision Tree) forces all three models.
- Alignment: good structurally (three techniques forced), **but the `SVM MSE > 1` check is a bug — it gates nothing.**
- Value: good (multi-model). Changes: **fix the SVM MSE threshold** to a real upper bound; note the assignment really covers three sessions.

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 18: Support Vector Regression

##### Support Vector Regression
`unit_id: abcebe4a-ef8f-4d62-a081-41ea5724eef4` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: af0e3a8c-5558-4ff8-92e6-7411e629ac73`

**Teaches:** Support Vector Regression

_Flow:_ 1) Recap the regression setup and introduce SVR 2) Explain fitting a best-fit line within an epsilon-tube margin 3) Contrast SVR's margin focus with minimizing error for all points 4) Note real-world applications of SVR

_Outcomes (depth):_ Describe how SVR fits a line within an epsilon-tube margin [Explained]; Distinguish SVR's margin-based objective from minimizing error on all points [Explained]; Identify real-world applications suited to SVR [Introduced]

_Key takeaways:_ (1) SVR predicts a continuous target by fitting a best-fit line, similar to linear regression. (2) Unlike ordinary regression, SVR fits a line within a tolerance margin called the epsilon-tube instead of minimizing error for every point. (3) SVR applies to problems such as stock price prediction, housing price estimation, and demand forecasting.

**Taught how (4 patterns):** Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Real-world motivation with named domains; Recap bridge from prior session

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Support Vector Regression Implementation
`unit_id: 9d214c9d-8e2a-4be7-995e-4499ae289802` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: af0e3a8c-5558-4ff8-92e6-7411e629ac73`

**Teaches:** Data Visualization, Feature Scaling, HyperParameter Tuning, Regression Metrics (R2/MSE/RMSE/MAE), Standardization (Z-score Scaling), Support Vector Regression, Train-Test Split

_Flow:_ 1) Define the sales-prediction problem and load the advertising data 2) Inspect data, analyze outliers, and explore feature relationships 3) Split the data and standardize features fitting the scaler on training data 4) Train an SVR model with an RBF kernel and evaluate with MSE 5) Visualize actual versus predicted values 6) Tune C, epsilon, kernel, and gamma with GridSearchCV 7) Visualize how C, epsilon, and weight magnitude affect the fit and margin 8) Review SVR pros and cons

_Outcomes (depth):_ Train and evaluate an SVR model on a scaled sales dataset [Applied]; Standardize features before fitting SVR to address its scale sensitivity [Applied]; Tune SVR hyperparameters C, epsilon, kernel, and gamma with GridSearchCV [Applied]; Interpret how C and epsilon change the regression line and tube width [Deep-dive]; Weigh the pros and cons of SVR for a modeling task [Explained]

_Key takeaways:_ (1) Applying SVR requires feature scaling, since the algorithm is sensitive to feature magnitudes. (2) An SVR model is defined by its kernel and the hyperparameters C and epsilon, commonly tuned with grid search and cross-validation. (3) The C parameter governs the trade-off between a flatter, better-generalizing line and a steeper line that overfits. (4) The epsilon parameter sets the width of the tube within which prediction errors are tolerated. (5) SVR handles high-dimensional and non-linear data through kernels and is robust to outliers via its margin, but is computationally costly and scaling-sensitive.

**Taught how (8 patterns):** Agenda-first, Key-Takeaways-last framing; Hyperparameter sweep to reveal effect and trade-offs; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 19: Decision Tree Regression

##### Decision Tree Regression
`unit_id: daf48d8a-ffb0-409c-97e3-4e474c382c71` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 81046be0-0708-453e-9812-2bf297e2a884`

**Teaches:** Decision Tree Regression

_Flow:_ 1) Recap decision tree structure (root, internal nodes, branches, leaves) 2) Introduce decision tree regression predicting continuous values by splitting 3) Explain selecting candidate split points as midpoints of sorted feature values 4) Compute MSE and the weighted average error for each candidate split 5) Select the feature and split point with the lowest weighted MSE 6) Create left and right child nodes and recurse until a stopping condition 7) Assign each leaf the average target value of its samples 8) Trace how a new point is routed down the tree to a leaf prediction

_Outcomes (depth):_ Explain how a regression tree splits data to predict continuous values [Explained]; Identify candidate split points as midpoints of sorted feature values [Explained]; Compute the weighted average MSE of a split to evaluate it [Applied]; Assign leaf predictions as the average target of the samples in the leaf [Explained]; Select the best split by minimizing weighted MSE across features and split points [Deep-dive]; Trace a new sample down the tree to produce a prediction [Applied]

_Key takeaways:_ (1) A decision tree regressor predicts continuous values by recursively splitting data into subsets that reduce prediction error. (2) Candidate split points are the midpoints between sorted adjacent feature values. (3) The best split is chosen by minimizing the weighted average of child-node MSE across all features and split points. (4) Splitting recurses until a stopping condition is met, and each leaf predicts the average target of its samples. (5) A prediction is produced by routing a new sample down the tree using split comparisons until it reaches a leaf.

**Taught how (6 patterns):** Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; Paired theory-plus-implementation with live-demo cues; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Worked example computed step by step

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Decision Tree Regression Implementation
`unit_id: 6bddc7e4-00e5-4823-a654-9aa239178e15` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 81046be0-0708-453e-9812-2bf297e2a884`

**Teaches:** Data Visualization, Decision Tree Regression, HyperParameter Tuning, Regression Metrics (R2/MSE/RMSE/MAE), Train-Test Split

_Flow:_ 1) Define the sales-prediction problem and load the advertising data 2) Inspect data, analyze outliers, and explore feature relationships 3) Split the data into training and test sets 4) Train a DecisionTreeRegressor and evaluate with MSE 5) Visualize actual versus predicted values 6) Tune max depth and minimum split and leaf sizes with GridSearchCV 7) Review decision tree regression pros and cons

_Outcomes (depth):_ Train and evaluate a DecisionTreeRegressor on a sales dataset [Applied]; Evaluate the model using MSE and actual-versus-predicted plots [Applied]; Tune tree-complexity hyperparameters with GridSearchCV to reduce overfitting [Applied]; Explain why decision trees do not require feature scaling [Explained]; Assess the pros and cons of decision tree regression [Explained]

_Key takeaways:_ (1) A decision tree regressor can be trained directly on unscaled features, since it does not require feature scaling. (2) Model quality is assessed with mean squared error and by comparing actual against predicted values. (3) Tree-complexity controls such as max depth, minimum samples to split, and minimum samples per leaf can be tuned with grid search to curb overfitting. (4) Decision trees are easy to interpret and capture non-linear relationships but are prone to overfitting, unstable to small data changes, and biased toward high-cardinality features.

**Taught how (6 patterns):** Agenda-first, Key-Takeaways-last framing; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Problem statement first (business framing); Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 20: Capstone Project (Life Expectancy Prediction)

##### Problem Statement
`unit_id: 34540853-2fd4-4182-aef7-1cc74b742684` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 63b6a671-a072-4a1b-b487-90b0097ce31c`

**Teaches:** _none tagged_

_Flow:_ 1) Frame life expectancy prediction as a regression problem 2) Generate hypotheses about factors that influence the target 3) Review the dataset's feature dictionary and units 4) Import libraries and load the dataset 5) Inspect shape, columns, dtypes, and summary statistics 6) Flag implausible minimum and maximum values for later cleaning

_Outcomes (depth):_ Frame a real-world prediction task as a supervised regression problem [Explained]; Generate hypotheses about factors influencing the target before analysis [Explained]; Interpret a dataset's feature dictionary and the meaning of each variable [Explained]; Load a CSV dataset and inspect its shape, columns, and dtypes with pandas [Applied]; Summarize distributions with describe() and flag implausible extreme values [Applied]

_Key takeaways:_ (1) Life expectancy prediction is framed as a supervised regression problem driven by socio-economic, health, and environmental indicators. (2) Hypothesis generation lists candidate drivers such as GDP, immunization coverage, education, disease prevalence, population density, and adult mortality before touching the data. (3) The dataset spans countries across 2000 to 2015 with a Developed or Developing status flag and numeric health and economic features. (4) Summary statistics expose implausible extremes such as very high infant-death and Measles maxima and a minimum BMI of 1, signalling likely errors or outliers.

**Taught how (6 patterns):** Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; Hypothesis generation before analysis; ML workflow skeleton in implementation notebooks; Problem statement first (business framing); Real-world motivation with named domains

**Assessed by — Capstone Project 1 — Regression (Trip Duration)** (HARD · RMSE ≤ 900 s · style 5)
- Tests: ~39k-row submission, positive finite predictions, RMSE ≤ 900 seconds.
- Alignment: good — realistic scale and an absolute error bar. Value: high. Changes: none urgent.

**Supporting content:** 1 reading, 1 code

---

##### EDA
`unit_id: 2c8b680d-0d85-49e8-a832-84aff5d03259` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 63b6a671-a072-4a1b-b487-90b0097ce31c`

**Teaches:** Data Visualization, Exploratory Data Analysis (EDA)

_Flow:_ 1) Standardize column names 2) Check for duplicate rows 3) Quantify missing values by count and percentage 4) Run univariate analysis with histograms and boxplots 5) Inspect category counts for Country and Status 6) Run bivariate analysis against the target with barplots and scatterplots

_Outcomes (depth):_ Clean and standardize column names before analysis [Applied]; Quantify missing values as raw counts and percentages [Applied]; Distinguish univariate from bivariate analysis [Explained]; Visualize each numeric feature with paired histograms and boxplots [Applied]; Explore feature-target relationships using scatterplots and grouped barplots [Applied]

_Key takeaways:_ (1) Univariate analysis summarizes one variable at a time through its distribution shape, central tendency, and spread. (2) Bivariate analysis studies the relationship between two variables to surface correlations and trends. (3) Pairing histograms with KDE and boxplots reveals skew and outliers for each numeric feature. (4) Scatterplots of features against life expectancy expose the direction and strength of relationships, while a Developed versus Developing barplot contrasts group means. (5) Missing-value counts and percentages are computed early to plan later imputation.

**Taught how (8 patterns):** Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Problem statement first (business framing); Real-world motivation with named domains; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments

**Supporting content:** 1 reading, 1 code

---

##### Missing Values and Outliers Treatment
`unit_id: c580a020-cade-4b9d-bb3d-c1688649bf13` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 63b6a671-a072-4a1b-b487-90b0097ce31c`

**Teaches:** Data Visualization, Missing Values & Outlier Treatment

_Flow:_ 1) Match each column to an imputation strategy by its nature 2) Impute continuous columns with mean and vaccination columns with mode using SimpleImputer 3) Drop rows missing the target value 4) Interpolate time-varying GDP and Population linearly 5) Fill schooling and income composition with group medians by country then status 6) Detect outliers with the IQR rule and replace them with the column mean 7) Re-check distributions with boxplots

_Outcomes (depth):_ Select an imputation method appropriate to each column's characteristics [Deep-dive]; Impute missing values using scikit-learn SimpleImputer with mean and mode strategies [Applied]; Drop records that are missing the target variable [Applied]; Interpolate time-series features linearly to fill gaps [Applied]; Fill grouped features with per-group medians using groupby transform [Applied]; Detect and treat outliers using the IQR method [Applied]

_Key takeaways:_ (1) Imputation strategy is chosen per column by its nature: central tendency for continuous features, mode for clustered percentages, interpolation for time series, and group statistics for country-linked features. (2) Rows missing the target variable are dropped rather than imputed, since fabricating the label would corrupt training. (3) Linear interpolation suits features that change gradually over years, such as GDP and Population. (4) Grouping by country and then by status lets fill values reflect a country's own level when imputing schooling and income composition. (5) The IQR rule flags values beyond 1.5 times the interquartile range as outliers, here replaced with the column mean, with boxplots confirming the treatment.

**Taught how (7 patterns):** Imputation/cleaning strategy justified by feature characteristics; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Problem statement first (business framing); Real-world motivation with named domains; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments

**Supporting content:** 1 reading, 1 code

---

##### Feature Engineering - 1
`unit_id: ae4d661a-8367-4dd3-b5cc-6c165a5b44e4` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 63b6a671-a072-4a1b-b487-90b0097ce31c`

**Teaches:** Data Visualization, Feature Engineering, Missing Values & Outlier Treatment

_Flow:_ 1) Review categorical encoding techniques: label, one-hot, and target encoding 2) Binary-encode the Status column to 0 and 1 3) Target-encode high-cardinality Country by mean life expectancy 4) Visualize feature correlations with a heatmap

_Outcomes (depth):_ Compare label, one-hot, and target encoding for categorical features [Explained]; Compute a target-encoded value as the mean of the target within each category [Explained]; Binary-encode a two-category column to 0 and 1 [Applied]; Apply target encoding to a high-cardinality feature using group means [Applied]; Read a correlation heatmap to gauge feature-target relationships [Applied]

_Key takeaways:_ (1) Label encoding maps categories to integers, one-hot encoding creates a binary column per category, and target encoding replaces a category with the mean target value for that category. (2) Target encoding suits high-cardinality features such as Country, where one-hot encoding would create too many columns. (3) A two-category status field maps cleanly to 0 and 1 without one-hot expansion. (4) The target-encoded value of a category is the mean of the target over all rows in that category. (5) A correlation heatmap after encoding reveals which numeric features relate most strongly to life expectancy.

**Taught how (12 patterns):** Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; Imputation/cleaning strategy justified by feature characteristics; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Problem statement first (business framing); Small hand-computable toy dataset to teach mechanics; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments; Term-by-term decomposition of a formula; When-to-use guidance with concrete triggers; Worked example computed step by step

**Supporting content:** 1 reading, 1 code

---

##### Build Base ML model
`unit_id: e6acab55-2daa-4cd3-bdeb-413569251eb2` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 63b6a671-a072-4a1b-b487-90b0097ce31c`

**Teaches:** Data Visualization, Feature Scaling, Missing Values & Outlier Treatment, Regression Metrics (R2/MSE/RMSE/MAE), Standardization (Z-score Scaling), Train-Test Split

_Flow:_ 1) Split features and target and hold out a test set 2) Standardize features with StandardScaler fit on training data only 3) Train a Linear Regression baseline 4) Define an evaluation function for MSE, MAE, RMSE, and R2 5) Train KNN, SVR, and Decision Tree regressors 6) Plot train versus test MSE across k to study fit

_Outcomes (depth):_ Split data into train and test sets with a fixed random state [Applied]; Standardize features by fitting the scaler on training data only [Applied]; Train Linear Regression, KNN, SVR, and Decision Tree regressors with scikit-learn [Applied]; Evaluate regression models with MSE, MAE, RMSE, and R2 [Applied]; Diagnose bias and variance by plotting error against the number of neighbors [Applied]; Explain how a decision tree regressor splits data and predicts leaf averages [Explained]

_Key takeaways:_ (1) Building a baseline means splitting the data, scaling features, and training several simple regressors to compare. (2) StandardScaler is fit on the training set only and applied to the test set to prevent data leakage. (3) Regression models are evaluated with MSE, MAE, RMSE, and R2 rather than classification accuracy. (4) Linear Regression, KNN, SVR, and Decision Tree bring contrasting assumptions: linear fit, distance-based, margin-based, and recursive splits. (5) Sweeping KNN error across values of k illustrates the bias-variance trade-off, while a decision tree splits to minimize error and predicts the mean target at each leaf.

**Taught how (9 patterns):** Agenda-first, Key-Takeaways-last framing; Imputation/cleaning strategy justified by feature characteristics; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Problem statement first (business framing); Real-world motivation with named domains; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments

**Supporting content:** 1 reading, 1 code

---

##### Feature Engineering - 2
`unit_id: 5676068b-47c3-4928-a8d0-7ee89d574259` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 63b6a671-a072-4a1b-b487-90b0097ce31c`

**Teaches:** Data Visualization, Feature Engineering, Feature Scaling, Missing Values & Outlier Treatment, Regression Metrics (R2/MSE/RMSE/MAE), Standardization (Z-score Scaling), Train-Test Split

_Flow:_ 1) Introduce regularization as an overfitting control 2) Contrast L2 (Ridge) and L1 (Lasso) penalties 3) Fit Lasso and inspect the shrunken coefficients 4) Use coefficient behavior to guide feature selection 5) Drop the highly correlated Country feature 6) Retrain the models on the reduced feature set

_Outcomes (depth):_ Explain how regularization penalizes model complexity to reduce overfitting [Explained]; Contrast L1 (Lasso) and L2 (Ridge) penalties and their effect on coefficients [Deep-dive]; Fit a Lasso model and interpret its coefficients [Applied]; Use Lasso's coefficient-zeroing behavior to guide feature selection [Applied]; Drop a highly correlated feature and retrain the models on the reduced set [Applied]

_Key takeaways:_ (1) Regularization adds a penalty term to the loss function to discourage overly complex models and curb overfitting. (2) Ridge (L2) penalizes squared coefficients, shrinking them toward zero without eliminating features, which helps with correlated predictors. (3) Lasso (L1) penalizes absolute coefficients and can drive some to exactly zero, performing automatic feature selection. (4) The lambda hyperparameter controls the strength of regularization. (5) Removing a target-encoded feature that is highly correlated with the target tests whether the models still generalize without that dominant signal.

**Taught how (13 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Imputation/cleaning strategy justified by feature characteristics; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Problem statement first (business framing); Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments; Term-by-term decomposition of a formula; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 code

---

##### Build Final ML model
`unit_id: 686d0668-9e6e-4065-8dad-a0ce4c57ae3e` · `course_id: 7993c9c1-49e9-4119-b9e3-41d670025951` · `topic_id: 63b6a671-a072-4a1b-b487-90b0097ce31c`

**Teaches:** Data Visualization, Feature Scaling, HyperParameter Tuning, Missing Values & Outlier Treatment, Regression Metrics (R2/MSE/RMSE/MAE), Standardization (Z-score Scaling), Train-Test Split

_Flow:_ 1) Set up GridSearchCV with 5-fold cross-validation 2) Tune Decision Tree hyperparameters 3) Tune KNN hyperparameters 4) Tune SVR hyperparameters 5) Record best parameters and R2 for each model 6) Compare final model R2 scores in a bar chart

_Outcomes (depth):_ Tune model hyperparameters with GridSearchCV and cross-validation [Applied]; Define parameter grids for Decision Tree, KNN, and SVR models [Applied]; Extract the best parameters and estimator from a grid search [Applied]; Compare tuned models on R2 to select the final regressor [Applied]; Explain why cross-validated search generalizes better than tuning on one split [Explained]

_Key takeaways:_ (1) GridSearchCV searches a parameter grid with cross-validation to find the settings that minimize validation error. (2) Each model family exposes distinct hyperparameters worth tuning: tree depth and split limits, KNN neighbors, weights, and metric, and SVR kernel, C, and epsilon. (3) Cross-validation gives a more reliable estimate of generalization than a single split when selecting parameters. (4) Comparing tuned models on a common metric such as R2 identifies the strongest regressor for the task.

**Taught how (7 patterns):** Imputation/cleaning strategy justified by feature characteristics; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Problem statement first (business framing); Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments

**Supporting content:** 1 reading, 1 code

---

### Ensemble Learning

#### Module 21: Introduction to Ensemble Algorithms

##### Introduction to Ensemble Algorithms
`unit_id: 83421c90-225c-4400-8bed-483ee0923400` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 2132eae2-7f36-4b92-ad9b-65d3bef4219b`

**Teaches:** Ensemble Learning

_Flow:_ 1) What ensemble learning is: combining multiple base or weak learners for better performance 2) Why a single model falls short; combined models correct each other's mistakes (wisdom of the crowd) 3) Base learners and why decision trees are preferred: simple, fast, flexible, capture complex patterns 4) Types of ensembles: homogeneous (same model type) vs heterogeneous (different model types) 5) Voting: combine predictions of independently trained models for the final output 6) Stacking: add a meta-model that learns how to weight and combine base model predictions 7) Bagging: train same-type models in parallel on random subsets (bootstrap aggregation), e.g. Random Forest 8) Boosting: train weak learners sequentially so each corrects the previous model's errors 9) Key differences across bagging, boosting, voting, and stacking by training, combination, goal, and models

_Outcomes (depth):_ Explain how ensemble learning combines multiple models to make more accurate and stable predictions [Explained]; Distinguish homogeneous from heterogeneous ensembles by the type of base learners used [Explained]; Compare bagging, boosting, voting, and stacking across training style, combination method, and goal [Deep-dive]; Justify why decision trees are commonly chosen as base learners in ensembles [Explained]; Identify which ensemble method fits a goal such as reducing variance versus reducing bias [Explained]

_Key takeaways:_ (1) Ensemble learning combines multiple base learners so the group produces more accurate and more stable predictions than any single model alone. (2) Ensembles are homogeneous when all base learners are the same type (e.g. all decision trees) and heterogeneous when the base learners are different model types. (3) Bagging trains models in parallel on different random subsets to reduce variance, while boosting trains weak learners sequentially to correct errors and reduce bias. (4) Voting combines independently trained models by majority vote or averaging, whereas stacking adds a meta-model that learns how much to trust each base model. (5) Decision trees are the most common base learners because they are simple, fast, flexible, and can capture non-linear patterns.

**Taught how (8 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Paired theory-plus-implementation with live-demo cues; Recap bridge from prior session

**Supporting content:** 1 reading, 1 in_class_quiz

---

#### Module 22: Voting and Stacking

##### Voting
`unit_id: 6cd972fc-53ed-48fb-91f7-31abe15d6be2` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 3ab3aab0-9682-43ec-a7b0-1d9bc457ef1a`

**Teaches:** Ensemble Learning, Regression Metrics (R2/MSE/RMSE/MAE), Train-Test Split, Voting

_Flow:_ 1) Recap: voting combines predictions of models trained independently on the same data 2) Voting in classification: two ways, hard and soft 3) Hard voting: each model votes a class, majority class wins (worked example) 4) Soft voting: average predicted class probabilities, highest average wins (worked example) 5) Voting in regression: average the models' continuous outputs (worked example) 6) Implementation: train individual classifiers, compare accuracy, then build a hard VotingClassifier 7) Implementation: train individual regressors, compare error, then build a VotingRegressor by averaging

_Outcomes (depth):_ Distinguish hard voting, soft voting, and regression averaging as ways to combine model predictions [Explained]; Compute a final voting prediction by majority vote, averaged probabilities, or averaged outputs [Applied]; Build a VotingClassifier and VotingRegressor from diverse base models and compare them against the individual models [Applied]; Justify why aggregating diverse models yields more stable and often more accurate predictions [Deep-dive]

_Key takeaways:_ (1) Voting combines predictions from several independently trained models to produce a more stable final output. (2) Hard voting selects the class with the most votes, while soft voting averages predicted class probabilities and chooses the highest. (3) In regression, voting averages the numeric outputs of the base models since hard and soft voting do not apply. (4) Because diverse base models make different errors, the voting ensemble often outperforms any single model. (5) Choosing structurally different base models increases diversity and strengthens the ensemble.

**Taught how (15 patterns):** Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Comparison table contrasting methods; Definition callouts for terminology; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Paired theory-plus-implementation with live-demo cues; Problem statement first (business framing); Real-world motivation with named domains; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated; Worked example computed step by step

**Assessed by — Voting & Stacking (Vitamin Deficiency, 5-class)** (EASY · both ≥ 85% · 6 test cases · style 2)
- Tests: `voting_pred` + `stacking_pred` columns (forces both methods), ≥3 base models required, both accuracy > 85%.
- Alignment: **good** — structurally forces both ensemble techniques. Value: high. Changes: none urgent.

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

##### Stacking
`unit_id: 20a2bd75-b0bf-4a19-a930-30add4929ee7` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 3ab3aab0-9682-43ec-a7b0-1d9bc457ef1a`

**Teaches:** Ensemble Learning, Feature Scaling, Regression Metrics (R2/MSE/RMSE/MAE), Stacking, Standardization (Z-score Scaling), Train-Test Split

_Flow:_ 1) Recap: stacking extends voting by adding a meta-model 2) How the meta-model works: it takes base model predictions as inputs and learns how to combine and weight them 3) Choosing a meta learner: simple models such as Logistic Regression for classification and Ridge or Lasso for regression 4) Why simple meta-models are preferred: reduce overfitting, handle correlated predictions, stay interpretable 5) Implementation note: stacking uses mlxtend for flexible prediction handling that helps prevent data leakage 6) Implementation (classification): train KNN and Decision Tree, form meta-features from their probabilities, train a StackingClassifier with a Logistic Regression meta-model, compare accuracy 7) Implementation (regression): train KNN, Linear Regression, and Decision Tree, build a StackingRegressor with a Ridge meta-model, compare RMSE

_Outcomes (depth):_ Explain how a meta-model learns to combine the predictions of several base models in stacking [Explained]; Distinguish stacking from voting by contrasting a learned weighting against a fixed combination rule [Deep-dive]; Construct meta-features from base model predictions to serve as inputs to the meta-model [Applied]; Build a StackingClassifier and StackingRegressor with chosen base learners and a simple meta-model, then compare against the base models [Applied]; Justify choosing a simple meta-model to limit overfitting and handle correlated base predictions [Deep-dive]

_Key takeaways:_ (1) Stacking adds a meta-model that takes the base models' predictions as inputs and learns how to best combine them. (2) Unlike voting's fixed averaging or majority rule, the meta-model learns from data how much to trust and weight each base model. (3) The base models' outputs, such as predicted class probabilities, become the input features for the meta-model. (4) Simple meta-models like Logistic Regression, Ridge, or Lasso are preferred because they reduce overfitting and handle correlated predictions well. (5) Stacking leverages the complementary strengths of different base models to improve overall prediction performance.

**Taught how (14 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Expose the fitted model's internals to demystify the abstraction; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Paired theory-plus-implementation with live-demo cues; Problem statement first (business framing); Recap bridge from prior session; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 23: Bagging & Random Forest

##### Bagging
`unit_id: a2ab5ef7-40bd-4ebd-ae7a-4b6ca8e478ad` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 4641fecc-398c-4d1a-80df-2e5528115ec2`

**Teaches:** Bagging, Data Visualization, Ensemble Learning, Evaluation Metrics, Label Encoding, Missing Values & Outlier Treatment, Train-Test Split

_Flow:_ 1) Recap: bagging as an ensemble that reduces variance 2) How bagging works: bootstrap samples -> train a model per sample -> aggregate 3) Sampling with replacement (bootstrapping) and why it creates diversity 4) Instance-based vs attribute-based sampling 5) Aggregating predictions (majority vote for classification, average for regression); out-of-bag validation 6) Implementation: telecom churn - encode features, inspect correlations, train a BaggingClassifier, evaluate

_Outcomes (depth):_ Explain how bagging reduces variance through bootstrap sampling and aggregation [Explained]; Describe sampling with replacement and why it produces diverse base models [Explained]; Build and evaluate a BaggingClassifier on a real dataset [Applied]; Reason about why aggregating diverse models is more stable than a single model [Deep-dive]

_Key takeaways:_ (1) Bagging (Bootstrap Aggregation) reduces variance by training many models on different bootstrap samples and combining them. (2) Sampling with replacement makes each training subset different, which is what gives the ensemble its diversity. (3) Predictions are aggregated by majority vote (classification) or averaging (regression). (4) Out-of-bag samples give built-in validation without a separate hold-out set.

**Taught how (18 patterns):** Active recall via posed questions; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Expose the fitted model's internals to demystify the abstraction; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Paired theory-plus-implementation with live-demo cues; Problem statement first (business framing); Real-world motivation with named domains; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers; Worked example computed step by step

**Assessed by — Bagging (Titanic)** (toughness unset · both > 85% · style 2)
- Tests: `Bagging_DT_survived` + `RF_survived` columns (forces a bagged DT and a random forest), both accuracy > 85%.
- Alignment: **good** — contrasts plain bagging vs random forest. Value: high. Changes: set a toughness label.

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

##### Random Forest
`unit_id: 275f25cb-aff3-426d-95fe-6e61fcc3f6ed` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 4641fecc-398c-4d1a-80df-2e5528115ec2`

**Teaches:** Bagging, Random Forest

_Flow:_ 1) Recap: bagging reduces variance by combining models trained on bootstrap samples 2) Random Forest as bagging with decision trees plus random feature selection at each split 3) Why randomly pick features: strong features dominate splits and make trees correlated 4) Credit risk example: using different feature combinations avoids over-reliance on one feature 5) Working steps: choose number of trees, create bootstrap samples, select a random feature subset per split, build trees 6) Make predictions across all trees and aggregate by majority vote (argmax) for classification 7) Aggregate by averaging tree outputs for regression 8) Variance as a strength: random feature selection decorrelates trees for better generalization 9) Compare Decision Tree, Bagging, and Random Forest by data, features at split, and diversity source

_Outcomes (depth):_ Explain how Random Forest extends bagging by adding random feature selection at each split [Explained]; Describe why restricting features per split decorrelates trees and strengthens the ensemble [Explained]; Compute a Random Forest prediction using majority vote for classification and averaging for regression [Applied]; Distinguish Decision Tree, Bagging, and Random Forest by training data, features at split, and diversity source [Deep-dive]

_Key takeaways:_ (1) Random Forest is bagging with decision trees plus random feature selection at each split, which decorrelates the trees. (2) Selecting a random subset of features at each split stops a strong feature from dominating every tree and increases diversity. (3) Classification predictions are aggregated by majority vote and regression predictions by averaging across the trees. (4) By reducing correlation between bagged trees, Random Forest lowers variance and improves generalization.

**Taught how (10 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Real-world motivation with named domains; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; Worked example computed step by step

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### AUC - ROC Curve
`unit_id: 0dbe0f80-ed0e-4c44-b116-ddffc69fa79e` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 4641fecc-398c-4d1a-80df-2e5528115ec2`

**Teaches:** AUC-ROC Curve, Evaluation Metrics

_Flow:_ 1) Recap confusion matrix: TP, FN, FP, TN 2) What AUC-ROC measures: how well a model separates classes across thresholds 3) True Positive Rate as TP / (TP + FN) 4) False Positive Rate as FP / (FP + TN) 5) Classification threshold converts probability to a class label and trades TPR against FPR 6) ROC curve: plot TPR against FPR for every threshold 7) AUC as the area under the ROC curve (1.0 perfect, 0.5 random) 8) Spam example: derive threshold points and compute AUC using the trapezoidal rule 9) Limitations on imbalanced data and the Precision-Recall curve as an alternative

_Outcomes (depth):_ Compute TPR and FPR from confusion matrix counts [Applied]; Explain how changing the classification threshold moves a point along the ROC curve [Explained]; Interpret an AUC value as a measure of class separation [Explained]; Compute AUC from discrete ROC points using the trapezoidal rule [Applied]; Justify preferring a Precision-Recall curve over ROC on imbalanced datasets [Deep-dive]

_Key takeaways:_ (1) TPR is the share of actual positives correctly caught and FPR is the share of actual negatives wrongly flagged as positive. (2) Lowering the classification threshold raises both TPR and FPR, so threshold choice trades sensitivity against false alarms. (3) The ROC curve plots TPR versus FPR across all thresholds, and AUC condenses it into a single number where 1.0 is perfect and 0.5 is random. (4) Thresholds are taken from the model's own predicted probabilities, and AUC is computed by summing the trapezoid areas between consecutive ROC points. (5) On highly imbalanced data ROC-AUC can look optimistic, so a Precision-Recall curve gives a more honest view of positive-class performance.

**Taught how (12 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Real-world motivation with named domains; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers; Worked example computed step by step

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Random Forest Implementation
`unit_id: f78375df-cbf0-40ee-bdae-11b9de49b7d5` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 4641fecc-398c-4d1a-80df-2e5528115ec2`

**Teaches:** AUC-ROC Curve, Class Imbalance Handling, Data Visualization, Evaluation Metrics, Label Encoding, Missing Values & Outlier Treatment, Random Forest, Train-Test Split

_Flow:_ 1) Define the problem: predict telecom customer churn 2) Load the dataset and inspect its structure 3) Preprocess: convert TotalCharges to numeric, label-encode categoricals, inspect a correlation heatmap 4) Split into train and test sets and train a RandomForestClassifier 5) Evaluate train vs test accuracy and visualize one tree from the forest 6) Assess the model with a confusion matrix and classification report 7) Plot the ROC curve (AUC = 0.84) and the PR curve (AP = 0.65) 8) Compare ROC-AUC with PR-AUC on imbalanced data 9) Review the pros and cons of Random Forest

_Outcomes (depth):_ Build and train a RandomForestClassifier on a real churn dataset [Applied]; Prepare data by encoding categorical features and examining feature correlations [Applied]; Evaluate a classifier using accuracy, a confusion matrix, a classification report, and ROC and PR curves [Applied]; Interpret ROC-AUC versus PR-AUC to judge performance on imbalanced data [Deep-dive]; Identify the strengths and weaknesses of Random Forest for a classification task [Explained]

_Key takeaways:_ (1) A Random Forest workflow runs from problem definition through preprocessing, training, and multi-metric evaluation. (2) Categorical features must be encoded to numeric before training, and a correlation heatmap reveals which features relate to churn. (3) Accuracy alone can hide weak minority-class performance; the churn model reached about 79% accuracy but only about 47% recall on churners. (4) ROC-AUC can look strong on imbalanced data while the PR curve exposes weaker positive-class performance. (5) Random Forest controls overfitting and exposes feature importance, but it is less interpretable and sensitive to class imbalance.

**Taught how (10 patterns):** Agenda-first, Key-Takeaways-last framing; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Problem statement first (business framing); Real-world motivation with named domains; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz

---

##### Feature Importance Techniques
`unit_id: 1b09e4fb-1ace-4f14-be9f-239898a2371f` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 4641fecc-398c-4d1a-80df-2e5528115ec2`

**Teaches:** Class Imbalance Handling, Data Visualization, Evaluation Metrics, Feature Importance, Label Encoding, Missing Values & Outlier Treatment, Train-Test Split

_Flow:_ 1) Recap Random Forest and motivate why feature importance matters 2) Random Forest feature importance from impurity reduction averaged across all trees 3) Play Tennis example: rank features within each tree, then combine into an overall ranking 4) Access scores via feature_importances_ and plot them, then select features above a threshold 5) Advantages and biases of RF importance: fast but favors high-cardinality features and does not imply causation 6) LOFO: remove one feature at a time, retrain, and measure the performance drop 7) Apply RFC threshold selection and LOFO in code on the churn dataset 8) Compare base, RFC-selected, and LOFO model accuracy 9) Choose between RFC importance and LOFO based on model type and computational cost

_Outcomes (depth):_ Explain how Random Forest derives feature importance from impurity reduction across trees [Explained]; Extract and rank feature importances with feature_importances_ and use a threshold to select features [Applied]; Implement LOFO by retraining after removing each feature and measuring the performance drop [Applied]; Distinguish RF importance from LOFO by approach, cost, and model applicability [Deep-dive]; Justify when to prefer RF importance versus LOFO for a given task [Deep-dive]

_Key takeaways:_ (1) Random Forest feature importance sums each feature's impurity reduction across its splits and averages it over all trees. (2) RF importance is fast but biased toward continuous or high-cardinality features and reflects correlation, not causation. (3) LOFO is model-agnostic: it removes one feature at a time, retrains, and ranks features by the resulting drop in performance. (4) LOFO measures direct performance impact but is expensive because the model must be retrained once for every feature. (5) Selecting features by a fixed importance threshold can discard features that are weak alone yet useful in combination.

**Taught how (15 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 24: Boosting

##### Boosting methods and Adaboost
`unit_id: cdb69b70-df3b-447c-8fd4-7072f37b4d51` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 58dfde4d-2962-442d-8c5f-82446c50a8ac`

**Teaches:** AdaBoost, Boosting

_Flow:_ 1) What boosting is: combining weak learners sequentially, unlike independent bagging models 2) Key concepts: weak learners, decision stumps, sequential learning, weighted samples 3) Types of boosting: AdaBoost, Gradient Boosting, XGBoost 4) How AdaBoost works: focus more on misclassified points each round, then weighted voting 5) Worked churn example: choosing the root split by information gain 6) Initialize equal sample weights and train the first weak learner (decision stump) 7) Compute weighted error and derive the learner weight alpha 8) Update sample weights for misclassified vs correct points, then normalize 9) Repeat rounds and combine learners into a final weighted score, sigmoid to probability, threshold to class 10) Advantages and disadvantages, including sensitivity to noise and overfitting from too many rounds

_Outcomes (depth):_ Explain how boosting differs from bagging by learning sequentially from previous errors [Explained]; Describe how AdaBoost reweights misclassified samples and weights learners by alpha [Explained]; Compute a weak learner's weighted error and its alpha, then update and normalize sample weights [Applied]; Justify why decision stumps only need to beat random guessing for AdaBoost to work [Deep-dive]; Identify AdaBoost's sensitivity to noise and the overfitting risk from too many iterations [Explained]

_Key takeaways:_ (1) Boosting builds models sequentially so each learner corrects the errors of the previous ones, unlike bagging where models are trained independently. (2) AdaBoost typically uses decision stumps as weak learners and increases the weight of misclassified samples so later learners focus on the hard cases. (3) Each weak learner receives an importance weight alpha computed from its weighted error, so more accurate learners have more say in the final prediction. (4) The final AdaBoost prediction is a weighted sum of learner outputs, which can be passed through a sigmoid and thresholded at 0.5 to produce a class. (5) AdaBoost is sensitive to noisy data and outliers, and the number of iterations must be tuned to avoid overfitting.

**Taught how (12 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Flag optional or simplified mathematical depth; Real-world motivation with named domains; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; When-it-fails honesty / limitations stated; Worked example computed step by step

**Assessed by — Boosting (Heart Disease)** (EASY · accuracy > 80% · style 5/6)
- Tests: `submission_df`, beat 80%. Suggests XGBClassifier but doesn't verify boosting.
- Alignment: partial — reasonable threshold; boosting not enforced. Value: moderate. Changes: verify a boosting estimator.

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Gradient Boosting Regression
`unit_id: 7b383f6b-808d-47fb-bec4-1a2bac38f931` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 58dfde4d-2962-442d-8c5f-82446c50a8ac`

**Teaches:** Gradient Boosting

_Flow:_ 1) Recap of boosting and AdaBoost; contrast with Gradient Boosting's loss-optimizing approach 2) Key components: loss function, weak learners fitting residuals, additive model 3) Step 1: initialize the model with the mean of the target 4) Step 2: compute residuals as actual minus prediction 5) Step 3: train a tree to predict residuals, not the target 6) Recognize overfitting when a tree perfectly fits residuals 7) Introduce a learning rate to shrink each tree's contribution 8) Update predictions, recompute smaller residuals, and train the next tree 9) General prediction formula: mean plus learning-rate-scaled tree outputs 10) Combine trees on a test input and interpret why many small corrections generalize better

_Outcomes (depth):_ Explain how Gradient Boosting fits successive trees to residuals to reduce loss [Explained]; Compute residuals, apply a learning rate, and update predictions across boosting rounds [Applied]; Combine the initial mean and scaled tree outputs to predict for a new input [Applied]; Justify how the learning rate controls overfitting and the speed of learning [Deep-dive]; Distinguish Gradient Boosting from AdaBoost by its residual-fitting rather than sample reweighting [Explained]

_Key takeaways:_ (1) Gradient Boosting adds weak learners sequentially to minimize a loss function, starting from the mean of the target for squared error loss. (2) Each tree is trained on the residuals of the current model rather than on the original target values. (3) A learning rate shrinks each tree's contribution, reducing overfitting and leaving room for later trees to correct errors gradually. (4) The final prediction is the initial mean plus the sum of learning-rate-scaled tree outputs across all trees. (5) Residuals shrink with each added tree, and combining many weak learners produces a strong model that generalizes better than a single deep tree.

**Taught how (13 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Flag optional or simplified mathematical depth; Paired theory-plus-implementation with live-demo cues; Preempt a look-alike confusion with explicit contrast; Real-world motivation with named domains; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; When-it-fails honesty / limitations stated; Worked example computed step by step

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Gradient Boosting Classification
`unit_id: bc8ec92e-65cc-4733-bf1d-739956e45b4e` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 58dfde4d-2962-442d-8c5f-82446c50a8ac`

**Teaches:** Gradient Boosting

_Flow:_ 1) Recap of Gradient Boosting regression and how classification differs only in loss and output interpretation 2) Why a Decision Tree Regressor is the base learner: gradients are continuous corrections 3) Step 1: build the initial model as a constant log-odds value from class proportions 4) Convert the log-odds score to a probability using the sigmoid function 5) Step 2: compute pseudo-residuals as actual label minus predicted probability 6) Step 3: fit a regression tree to the residuals and compute leaf values with the log-loss formula 7) Step 4: update each point's running log-odds score using the learning rate 8) Recompute probabilities and residuals, then repeat with more trees 9) Predict on unseen data: accumulate scores, apply sigmoid, threshold at 0.5 10) Advantages, disadvantages, and how overfitting and slow training are controlled

_Outcomes (depth):_ Explain why Gradient Boosting classification uses a regression tree on continuous gradients [Explained]; Compute the initial log-odds, convert it to a probability, and derive pseudo-residuals [Applied]; Compute leaf output values using the log-loss leaf formula and update log-odds scores [Applied]; Predict a class for an unseen point by accumulating scores, applying the sigmoid, and thresholding [Applied]; Justify how learning rate, tree depth, and early stopping control overfitting and training cost [Deep-dive]

_Key takeaways:_ (1) Gradient Boosting classification uses the same algorithm as regression, changing only the loss function and how the output is interpreted. (2) The base learner is a Decision Tree Regressor because gradients are continuous correction values, not class labels. (3) The initial model predicts a constant log-odds derived from class proportions, which the sigmoid converts to a probability. (4) Pseudo-residuals are computed as actual label minus predicted probability, and leaf values use the sum of residuals over the sum of previous_prob times one-minus-previous_prob. (5) Each point keeps a running score updated by learning-rate-scaled tree outputs, which is passed through a sigmoid and thresholded to produce the final class.

**Taught how (10 patterns):** Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; Preempt a look-alike confusion with explicit contrast; Real-world motivation with named domains; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers; Worked example computed step by step

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### XG boost
`unit_id: 68c7802b-b0bb-49c6-b200-78b77a33df48` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 58dfde4d-2962-442d-8c5f-82446c50a8ac`

**Teaches:** Gradient Boosting, XGBoost

_Flow:_ 1) Recap of Gradient Boosting; introduce XGBoost as an optimized, regularized implementation 2) Why XGBoost: adds regularization, efficient tree construction, and parallel processing for speed and scale 3) How it works: start from an initial mean prediction, compute residuals, fit a tree to them 4) Add the tree's contribution scaled by a learning rate and update predictions 5) Worked churn example: initial 0.5 prediction, residuals, tree corrections, updated predictions 6) Apply regularization (L1 and L2) as part of the tree-building objective to control complexity 7) Repeat by adding trees that correct earlier errors until convergence 8) Final refined predictions after several iterations 9) Key features, advantages, and disadvantages such as complexity and tuning effort

_Outcomes (depth):_ Explain how XGBoost extends Gradient Boosting with regularization, parallelism, and efficient tree building [Explained]; Trace an iteration of XGBoost from initial prediction through residuals to updated predictions [Applied]; Distinguish L1 and L2 regularization and their role in the tree-building objective [Explained]; Identify XGBoost strengths such as handling missing data and feature importance, and its overfitting and tuning trade-offs [Deep-dive]

_Key takeaways:_ (1) XGBoost is an optimized implementation of Gradient Boosting that adds regularization, efficient tree construction, and parallel processing. (2) Regularization (L1 and L2) is built into the objective while trees are constructed, controlling leaf weights and splits to prevent overfitting. (3) Like Gradient Boosting, XGBoost starts from an initial prediction, fits trees to residuals, and adds learning-rate-scaled corrections iteratively. (4) XGBoost can handle missing data directly, provides feature importance, and prunes trees, making it fast and accurate on structured tabular data. (5) Despite regularization XGBoost can still overfit with too many trees and requires careful hyperparameter tuning.

**Taught how (10 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Definition callouts for terminology; Flag optional or simplified mathematical depth; Real-world motivation with named domains; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers; Worked example computed step by step

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Boosting Implementation - Classification
`unit_id: 2fdde1a8-9fde-4b5c-bf57-fc32637e9285` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 58dfde4d-2962-442d-8c5f-82446c50a8ac`

**Teaches:** AUC-ROC Curve, Boosting, Data Visualization, Evaluation Metrics, Label Encoding, Missing Values & Outlier Treatment, One-Hot Encoding, Train-Test Split

_Flow:_ 1) Define the telecom churn problem and load the customer dataset 2) Transform data: set customerID as index, convert TotalCharges to numeric, impute with median 3) Identify categorical and numeric columns and label-encode categorical features 4) Inspect feature relationships with a correlation heatmap 5) Split features and target, then stratified train-test split 6) Build GradientBoostingClassifier, AdaBoostClassifier with a decision-stump base, and XGBClassifier 7) Compare train and test accuracy across the three models 8) Evaluate with classification reports and confusion matrices 9) Plot ROC curves and compare AUC values across models

_Outcomes (depth):_ Build a preprocessing workflow that encodes categorical features and handles missing numeric values for churn data [Applied]; Train GradientBoostingClassifier, AdaBoostClassifier, and XGBClassifier on the same dataset [Applied]; Evaluate and compare boosting classifiers using accuracy, classification reports, confusion matrices, and ROC-AUC [Applied]; Interpret overlapping ROC curves to judge relative model performance [Explained]

_Key takeaways:_ (1) Boosting classifiers can be trained on a real telecom churn dataset after encoding categorical features and imputing missing numeric values. (2) AdaBoost uses a shallow decision tree (a stump) as its base estimator, while Gradient Boosting and XGBoost use their own tree defaults. (3) Comparing train and test accuracy helps assess generalization, and Gradient Boosting and XGBoost generalized well on unseen churn data. (4) Classification reports, confusion matrices, and ROC-AUC together give a fuller view of model performance than accuracy alone. (5) All three boosting models achieved comparable ROC-AUC (roughly 0.84 to 0.85), showing similar ability to separate churn from non-churn customers.

**Taught how (9 patterns):** Agenda-first, Key-Takeaways-last framing; Comparison table contrasting methods; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Problem statement first (business framing); Real-world motivation with named domains; Teaching code carries inline explanatory comments

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz

---

##### Boosting Implementation - Regression
`unit_id: 7c098fa1-496a-4ccc-85de-456c22714a4b` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 58dfde4d-2962-442d-8c5f-82446c50a8ac`

**Teaches:** Boosting, Data Visualization, Regression Metrics (R2/MSE/RMSE/MAE), Train-Test Split

_Flow:_ 1) Load the advertising dataset with TV, Radio, and Newspaper spend to predict Sales 2) Transform data: drop the unnamed index column and confirm all features are numeric 3) Inspect feature relationships with a correlation heatmap 4) Split features and target into training and testing sets 5) Train GradientBoostingRegressor, AdaBoostRegressor, and XGBRegressor 6) Evaluate models with R-squared, MAE, and RMSE 7) Plot actual vs predicted sales scatter plots against a perfect-prediction line 8) Compare models and conclude which fits best

_Outcomes (depth):_ Build a regression workflow on numeric advertising data with a train-test split [Applied]; Train GradientBoostingRegressor, AdaBoostRegressor, and XGBRegressor on the same dataset [Applied]; Evaluate regressors using R-squared, MAE, and RMSE and compare their errors [Applied]; Interpret actual vs predicted scatter plots to judge which model fits best [Explained]

_Key takeaways:_ (1) Boosting regressors can predict a continuous target such as sales from numeric advertising-spend features without any encoding. (2) Regression models are evaluated with metrics like R-squared, MAE, and RMSE rather than accuracy. (3) Actual vs predicted scatter plots against a diagonal reference line visually reveal how closely predictions match true values. (4) Gradient Boosting fit this advertising data best with the lowest error and highest R-squared, while AdaBoost had the highest error. (5) XGBoost also performed strongly and is well suited to large and complex datasets due to its efficiency and scalability.

**Taught how (10 patterns):** Agenda-first, Key-Takeaways-last framing; Comparison table contrasting methods; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Problem statement first (business framing); Real-world motivation with named domains; Teaching code carries inline explanatory comments; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz

---

#### Module 25: Capstone Project (Credit card fraud Detection

##### Problem Statement
`unit_id: 3879b76b-bdaf-4a97-aaf0-e7c53807fe76` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 253b9675-48cb-45d1-b506-1f1263c81935`

**Teaches:** _none tagged_

_Flow:_ 1) Frame credit card fraud detection as a rare-event problem 2) State the goal: experiment with multiple ensemble models on imbalanced data 3) Understand the dataset (European cardholder transactions over two days) 4) Explain the PCA-anonymized features V1-V28, plus Time, Amount, and Class 5) Describe how PCA blends original features into transformed components 6) Justify why models can learn from non-interpretable PCA features 7) Explain why ensemble models suit this problem 8) Outline the project workflow from problem statement to model comparison

_Outcomes (depth):_ Describe the credit card fraud detection problem and why rare fraud makes it challenging [Explained]; Explain how PCA transforms original transaction features into anonymized components [Explained]; Justify why machine learning models can learn effectively from non-interpretable PCA features [Explained]; Explain why ensemble methods are well suited to imbalanced fraud detection [Explained]; Outline the end-to-end project workflow from problem statement to model comparison [Introduced]

_Key takeaways:_ (1) Credit card fraud detection is a rare-event classification problem where fraudulent transactions form a tiny minority of all transactions. (2) Features V1 to V28 are PCA-transformed to anonymize sensitive data, while Time and Amount remain untransformed and Class is the fraud label. (3) PCA's primary purpose is dimensionality reduction; the anonymization of features is a by-product, not its intended use. (4) Machine learning models do not need human-interpretable features and can learn the regions of PCA space where fraud tends to appear. (5) Ensemble methods suit this problem because they capture complex nonlinear boundaries and handle class imbalance better than single models.

**Taught how (7 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; ML workflow skeleton in implementation notebooks; Problem statement first (business framing); Real-world motivation with named domains; When-to-use guidance with concrete triggers

**Assessed by — Capstone — Ensemble (Earthquake Damage, 3-class)** (HARD · accuracy ≥ 70% · style 5)
- Tests: multi-class `damage_grade` (1/2/3), ≥70%. Alignment: good (genuine multi-class on real data). Value: high.

**Supporting content:** 1 reading

---

##### EDA
`unit_id: e577bba1-ac9c-4a9e-a461-2420146857af` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 253b9675-48cb-45d1-b506-1f1263c81935`

**Teaches:** Data Visualization, Exploratory Data Analysis (EDA)

_Flow:_ 1) Load the dataset and inspect its structure 2) Check for missing values 3) Univariate analysis: summarize transaction Amount for fraud vs normal 4) Bivariate analysis: compare amount distributions across classes 5) Bivariate analysis: plot Time vs Amount by class 6) Quantify the class imbalance (fraud percentage and ratio) 7) Visualize the class distribution with a bar plot

_Outcomes (depth):_ Load and inspect a transaction dataset and check it for missing values [Applied]; Perform univariate and bivariate analysis to compare fraud and normal transactions [Applied]; Quantify and visualize the degree of class imbalance in a dataset [Applied]; Interpret which features carry a useful fraud signal and which do not [Applied]; Explain why class imbalance makes accuracy a misleading metric [Explained]

_Key takeaways:_ (1) Univariate analysis summarizes a single variable's distribution, while bivariate analysis studies the relationship between two variables. (2) Fraud transactions concentrate at small amounts whereas normal transactions span a much wider range, making Amount a meaningful fraud signal. (3) The Time variable only records seconds elapsed and shows no clear fraud pattern, so it is a weak predictor. (4) The dataset is extremely imbalanced, with fraud making up only about 0.17% of transactions, so high accuracy can be achieved by ignoring fraud entirely. (5) Ensemble models are well suited to imbalanced data because they reduce bias toward the majority class and focus on minority cases.

**Taught how (9 patterns):** Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Paired theory-plus-implementation with live-demo cues; Real-world motivation with named domains; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 code

---

##### Model Building 1
`unit_id: 2791df44-9109-45bd-8be2-85134104b060` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 253b9675-48cb-45d1-b506-1f1263c81935`

**Teaches:** AUC-ROC Curve, Class Imbalance Handling, Data Visualization, Evaluation Metrics, Feature Scaling, Standardization (Z-score Scaling), Train-Test Split

_Flow:_ 1) Explain why accuracy is misleading under class imbalance 2) Select fraud-focused metrics: precision, recall, F1, and ROC-AUC 3) Split data into train and test sets with stratification 4) Scale Time and Amount for the linear baseline model 5) Contrast algorithm-level (class weights) and data-level ways to handle imbalance 6) Build and evaluate a class-weighted Logistic Regression baseline 7) Build and evaluate a class-weighted Random Forest 8) Compare the baseline and first ensemble on the precision-recall trade-off

_Outcomes (depth):_ Explain why accuracy is misleading for imbalanced classification and choose appropriate metrics [Explained]; Split data with stratification and scale features for a linear model [Applied]; Distinguish algorithm-level from data-level approaches to class imbalance [Explained]; Build and evaluate a class-weighted Logistic Regression baseline [Applied]; Build and evaluate a class-weighted Random Forest and compare it to the baseline [Applied]; Interpret the precision-recall trade-off between the baseline and the ensemble model [Applied]

_Key takeaways:_ (1) Under extreme class imbalance, accuracy is misleading, so precision, recall, F1, and ROC-AUC are used to measure how well fraud is detected. (2) Class imbalance can be handled at the algorithm level with class weights or at the data level by resampling the training data. (3) Setting class_weight to balanced shifts the decision boundary toward the minority class, which raises recall. (4) The class-weighted Logistic Regression baseline achieves high recall but very low precision, producing many false alarms. (5) Random Forest, the first ensemble model, learns nonlinear boundaries and sharply improves precision over the linear baseline, with a slight drop in recall. (6) Tree-based models do not require feature scaling because they compare values rather than their magnitudes.

**Taught how (13 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Real-world motivation with named domains; Teaching code carries inline explanatory comments; Train/test hygiene to prevent data leakage; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code

---

##### Model Building 2
`unit_id: bdc0c7a5-0c7b-450b-befb-b07252d8c005` · `course_id: 453ae28a-0fd5-42cc-94b3-8705d3018d59` · `topic_id: 253b9675-48cb-45d1-b506-1f1263c81935`

**Teaches:** AUC-ROC Curve, Class Imbalance Handling, Data Visualization, Evaluation Metrics, Feature Scaling, Standardization (Z-score Scaling), Train-Test Split

_Flow:_ 1) State the goal: apply data-level balancing to improve fraud detection 2) Explain data-level balancing and contrast oversampling with undersampling 3) Apply random oversampling to the training data only to avoid leakage 4) Train and evaluate a Random Forest on the balanced data 5) Train and evaluate XGBoost using scale_pos_weight on the imbalanced data 6) Compare all models with ROC and precision-recall curves 7) Tune the decision threshold to control the precision-recall balance

_Outcomes (depth):_ Explain data-level balancing and contrast oversampling with undersampling [Explained]; Apply random oversampling to training data without leaking into the test set [Applied]; Build and evaluate a Random Forest trained on oversampled data [Applied]; Build and evaluate an XGBoost model that uses scale_pos_weight for imbalance [Applied]; Compare models with ROC and precision-recall curves and interpret the results [Applied]; Tune the decision threshold to control the precision-recall trade-off [Applied]

_Key takeaways:_ (1) Data-level balancing adjusts the class distribution before training; random oversampling duplicates minority fraud samples to balance the data. (2) Oversampling is applied only to the training data while the test set is left untouched, preventing data leakage and inflated metrics. (3) After oversampling, Random Forest no longer needs class weights because each tree sees enough fraud samples, but duplicating samples raises the risk of overfitting. (4) XGBoost is a boosting ensemble that builds trees sequentially and handles imbalance internally through scale_pos_weight, giving the strongest balance of precision and recall. (5) Precision-recall curves are more informative than ROC for imbalanced data, and the decision threshold can be tuned to trade recall for precision without retraining.

**Taught how (16 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Paired theory-plus-implementation with live-demo cues; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Real-world motivation with named domains; Teaching code carries inline explanatory comments; Train/test hygiene to prevent data leakage; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 code

---

### Unsupervised Learning

#### Module 26: Introduction to Unsupervised Learning

##### Introduction to Unsupervised Learning
`unit_id: 2285a57a-f498-4625-b4a8-87f2509f1065` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 4b52ed88-7747-4df0-9815-290f7aee9d98`

**Teaches:** Clustering, Unsupervised Learning

_Flow:_ 1) Recap supervised learning: labeled data, classification and regression 2) Motivate unsupervised learning when correct outputs are unknown 3) Define unsupervised learning on unlabeled data 4) Introduce three techniques: clustering, association, dimensionality reduction 5) Explain clustering with the Meta ad-audience example and applications 6) Explain association via market-basket IF-THEN rules and applications 7) Explain dimensionality reduction: keep key features, drop redundancy and noise 8) Compare supervised and unsupervised learning side by side

_Outcomes (depth):_ Distinguish supervised from unsupervised learning by their data and goals [Explained]; Describe the three unsupervised learning techniques and their use cases [Explained]; Identify whether clustering, association, or dimensionality reduction fits a given problem [Explained]; Explain how association rules capture item relationships as IF-THEN patterns [Explained]; Recognize dimensionality reduction as a way to simplify high-feature data [Introduced]

_Key takeaways:_ (1) Unsupervised learning finds hidden patterns and structure in unlabeled data with no predefined output. (2) Its three main techniques are clustering, association, and dimensionality reduction. (3) Clustering groups similar data points using distance or similarity in the feature space. (4) Association discovers relationships between items and expresses them as IF-THEN rules, as in market-basket analysis. (5) Dimensionality reduction simplifies data by keeping the most important features and removing correlated or noisy ones.

**Taught how (6 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Comparison table contrasting methods; Definition callouts for terminology; Real-world motivation with named domains; Recap bridge from prior session

**Supporting content:** 1 reading, 1 in_class_quiz

---

#### Module 27: K-Means Clustering

##### K-Means Clustering
`unit_id: 8229cfaa-2acc-4006-8ee8-61c6b34d20e5` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 16e0f965-8ad9-4578-be48-8e03cc5bd50b`

**Teaches:** Clustering, K-Means Clustering

_Flow:_ 1) Introduce K-Means as an unsupervised clustering algorithm 2) State the goal: group similar points and represent each group by a centroid 3) List the five-step K-Means procedure 4) State the objective of minimizing WCSS (Inertia) 5) Work the customer income and spending example with K = 2 6) Select initial centroids for the example 7) Assign points to the nearest centroid using Euclidean distance 8) Recalculate centroids as the mean of assigned points 9) Repeat assignment and update until clusters stop changing

_Outcomes (depth):_ List the five steps of the K-Means algorithm [Explained]; Assign data points to the nearest centroid using Euclidean distance [Applied]; Recalculate cluster centroids as the mean of their points [Applied]; Explain why K-Means minimizes WCSS and forms spherical clusters [Explained]; Trace K-Means iterations to convergence on a small dataset [Applied]

_Key takeaways:_ (1) K-Means partitions unlabeled data into K clusters, each summarized by a centroid. (2) It alternates between assigning points to the nearest centroid and recomputing centroids as cluster means. (3) K-Means measures distance with Euclidean distance, which tends to produce round, spherical clusters. (4) Its objective is to minimize the within-cluster sum of squares (WCSS or Inertia). (5) The algorithm converges when point-to-cluster assignments no longer change.

**Taught how (8 patterns):** Agenda-first, Key-Takeaways-last framing; Algorithm stated as an explicit numbered step sequence; Definition callouts for terminology; Flag optional or simplified mathematical depth; Real-world motivation with named domains; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; Worked example computed step by step

**Assessed by — K-Means Clustering (Fitness)** (EASY · Silhouette ≥ 0.25 · style 4)
- Alignment: good (forces the K sweep taught), but 0.25 is a soft bar. Value: good. Changes: consider raising the Silhouette floor.

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### K-Means Clustering Part 2
`unit_id: 1786d611-c0ee-4836-a306-9ce43fa3cd1a` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 16e0f965-8ad9-4578-be48-8e03cc5bd50b`

**Teaches:** K-Means Clustering

_Flow:_ 1) Motivate systematic selection of K instead of manual choice 2) Introduce the elbow method and the silhouette method 3) Explain WCSS as a measure of cluster compactness 4) Plot the elbow curve and locate the elbow point 5) Introduce the silhouette method and its purpose 6) Define intra-cluster distance a and inter-cluster distance b 7) Present the silhouette score formula and its -1 to +1 range 8) Work the silhouette calculation for a sample point at k = 2 9) Average silhouette scores and compare k values to choose the best

_Outcomes (depth):_ Apply the elbow method to select K from a WCSS curve [Applied]; Compute a silhouette score from intra- and inter-cluster distances [Applied]; Interpret silhouette values across the -1 to +1 range [Explained]; Choose the optimal K by comparing average silhouette scores [Applied]; Explain when the silhouette method is preferred over the elbow method [Explained]

_Key takeaways:_ (1) The elbow method plots WCSS against K and picks the K where the improvement in WCSS slows sharply. (2) WCSS measures cluster compactness and always decreases as K increases. (3) The silhouette method judges how well each point fits its cluster using intra-cluster (a) and inter-cluster (b) distances. (4) The silhouette score equals (b - a) / max(a, b) and ranges from -1 to +1, with higher values meaning better separation. (5) The best K is the one with the highest average silhouette score, tested from k = 2 up to the square root of the number of points.

**Taught how (10 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Definition callouts for terminology; Metric-driven model selection, triangulated; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; When-it-fails honesty / limitations stated; Worked example computed step by step

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### K-Means Clustering Part 3
`unit_id: dcfec5f5-07f9-4ab3-a1fd-b743e01945fb` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 16e0f965-8ad9-4578-be48-8e03cc5bd50b`

**Teaches:** K-Means Clustering

_Flow:_ 1) Motivate careful initial centroid selection for stable clustering 2) Explain random initialization and its drawbacks 3) Explain multiple random initializations and choosing the best run 4) Introduce K-Means++ and its goal of spreading centroids apart 5) List the K-Means++ centroid-selection steps 6) Work the K-Means++ example: distances, squared distances, probabilities, next centroid 7) Introduce clustering evaluation metrics: Inertia, Dunn Index, Silhouette 8) Explain the Dunn Index as a ratio to be maximized 9) Summarize the advantages and challenges of K-Means

_Outcomes (depth):_ Explain why random centroid initialization can produce unstable clusters [Explained]; Describe the K-Means++ centroid selection procedure [Explained]; Compute selection probabilities from squared distances in K-Means++ [Applied]; Compare Inertia, Dunn Index, and silhouette score as evaluation metrics [Explained]; Identify the advantages and challenges of K-Means [Explained]

_Key takeaways:_ (1) Initial centroid selection strongly affects clustering quality, number of iterations, and stability. (2) Running K-Means several times with different random starts and keeping the lowest-WCSS result reduces bad initialization. (3) K-Means++ spreads initial centroids by choosing far-apart points using probabilities proportional to squared distance. (4) Clustering quality is measured by Inertia (compactness), the Dunn Index (separation vs compactness), and the silhouette score. (5) The Dunn Index is the minimum inter-cluster distance over the maximum intra-cluster distance and should be maximized.

**Taught how (11 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Algorithm stated as an explicit numbered step sequence; Build the concept by fixing the previous version's limitation; Definition callouts for terminology; Metric-driven model selection, triangulated; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; When-it-fails honesty / limitations stated; Worked example computed step by step

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### K-Means implementation
`unit_id: 00ee847c-ac2a-4b0a-9e5d-e5b062d93694` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 16e0f965-8ad9-4578-be48-8e03cc5bd50b`

**Teaches:** Elbow Method, Feature Scaling, K-Means Clustering, Silhouette Score, Standardization (Z-score Scaling)

_Flow:_ 1) Problem statement: segment mall customers for targeted marketing 2) Load and inspect the data (shape, info, describe, missing values) 3) Data cleaning: encode Gender, drop the ID column 4) Feature selection (Annual Income, Spending Score); PCA note for higher dimensions 5) Feature scaling with StandardScaler and why distance-based clustering needs it 6) Choosing K with the elbow method (inertia) 7) Build K-Means, evaluate clusters with the silhouette score 8) Pros and cons of K-Means

_Outcomes (depth):_ Prepare a dataset for clustering (clean, select features, scale) [Applied]; Build a K-Means model and choose K using the elbow method and silhouette score [Applied]; Justify why feature scaling is required for distance-based clustering [Explained]; Evaluate and interpret cluster quality with the silhouette score [Applied]; Recognise the limitations of K-Means (spherical assumption, sensitivity to K and scaling) [Explained]

_Key takeaways:_ (1) K-Means groups points into K clusters by distance to centroids, so features must be scaled first. (2) K must be chosen in advance; the elbow method and silhouette score help pick it. (3) The silhouette score measures how well-separated the clusters are. (4) K-Means is fast and simple but sensitive to initialization and scaling, and assumes roughly spherical clusters.

**Taught how (12 patterns):** Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; Flag optional or simplified mathematical depth; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Real-world motivation with named domains; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz

---

#### Module 28: Hierarchical Clustering

##### Hierarchical Clustering
`unit_id: 1761a9de-98d0-4927-a348-418a409d4eaf` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 1d627f49-f4e8-4da5-a979-26c6cb541d7e`

**Teaches:** Clustering, Hierarchical Clustering

_Flow:_ 1) Recap K-Means and motivate hierarchical clustering for nested or irregular data 2) Define hierarchical clustering using the fruit-weight merging example 3) Introduce agglomerative bottom-up and divisive top-down approaches 4) Detail the agglomerative steps from singletons to a dendrogram 5) Compare single, complete, average, and Ward linkage methods 6) Explain divisive clustering as top-down splitting 7) Read a dendrogram and pick clusters at the largest vertical gap 8) Contrast hierarchical clustering with K-Means

_Outcomes (depth):_ Explain how agglomerative clustering builds a hierarchy by merging clusters [Explained]; Compare single, complete, average, and Ward linkage methods [Deep-dive]; Read a dendrogram to determine the number of clusters [Applied]; Distinguish agglomerative from divisive clustering [Explained]; Contrast hierarchical clustering with K-Means on approach and output [Deep-dive]

_Key takeaways:_ (1) Hierarchical clustering builds a tree of clusters and does not require the number of clusters in advance. (2) Agglomerative clustering merges the closest clusters bottom-up, while divisive clustering splits top-down. (3) The linkage method (single, complete, average, Ward) defines how distance between clusters is measured and shapes the resulting clusters. (4) A dendrogram visualizes the merge hierarchy, where lower merge height means the groups are more similar. (5) The number of clusters is chosen by cutting the dendrogram at its largest vertical gap.

**Taught how (12 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Algorithm stated as an explicit numbered step sequence; Analogy or concrete scenario before formalism; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers

**Assessed by — Hierarchical Clustering (Passenger)** (EASY · Silhouette ≥ 0.40 · style 4)
- Alignment: **good** — forces linkage exploration. Value: high. Changes: none urgent.

**Supporting content:** 1 reading, 1 in_class_quiz

---

##### Hierarchical Clustering Implementation
`unit_id: f8e15cd5-e68d-4491-b2d4-04bff5b549d7` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 1d627f49-f4e8-4da5-a979-26c6cb541d7e`

**Teaches:** Hierarchical Clustering, Silhouette Score

_Flow:_ 1) Define the problem: cluster the non-convex Moons dataset and compare algorithms 2) Load and plot the Moons dataset 3) Build a K-Means model and visualize its clusters 4) Observe K-Means failing on the interlocking crescent shapes 5) Build a dendrogram with Ward linkage to inspect structure 6) Apply Agglomerative Clustering with single linkage 7) Visualize and confirm correct separation of the two moons 8) Compare single, complete, and Ward linkage results 9) Review hierarchical clustering pros and cons

_Outcomes (depth):_ Generate and visualize a non-convex dataset for clustering [Applied]; Build and interpret a dendrogram from a linkage matrix [Applied]; Apply Agglomerative Clustering with a chosen linkage method [Applied]; Compare linkage methods on non-convex data [Deep-dive]; Explain why single linkage separates the moons better than K-Means [Explained]; Summarize the pros and cons of hierarchical clustering [Explained]

_Key takeaways:_ (1) K-Means fails on non-convex shapes like interlocking moons because it assumes spherical clusters. (2) A dendrogram built from a linkage matrix helps decide the number of clusters from large vertical gaps. (3) Single linkage follows chain-like, elongated structures and correctly separates the moons. (4) The choice of linkage method strongly affects clustering quality, and complete and Ward linkage distort non-convex shapes. (5) Hierarchical clustering needs no preset K and is deterministic, but has high computational cost and is sensitive to noise and outliers.

**Taught how (9 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; ML workflow skeleton in implementation notebooks; Observe-and-interpret loop; Problem statement first (business framing); Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 29: DBSCAN

##### DBSCAN
`unit_id: e73d28de-af17-4a12-9ed4-8586b16a16b4` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: bec64ffc-c86b-418e-874a-cb85a2941110`

**Teaches:** Clustering, DBSCAN

_Flow:_ 1) Recap limits of K-Means and hierarchical clustering with irregular shapes and noise 2) Define DBSCAN as density-based clustering that detects noise 3) Explain how DBSCAN differs: arbitrary shapes, automatic noise, no preset k 4) Introduce the parameters epsilon and MinPts 5) Choose epsilon via the k-distance graph and set MinPts using D + 1 6) Classify points as core, border, or noise 7) Explain density-connectivity and when it breaks 8) Walk through DBSCAN forming clusters step by step 9) Review advantages, limitations, and comparison with other methods

_Outcomes (depth):_ Define DBSCAN and explain its density-based approach to clustering [Explained]; Distinguish core, border, and noise points using epsilon and MinPts [Explained]; Explain how the k-distance graph guides epsilon selection [Explained]; Trace how density-connectivity expands a cluster [Explained]; Compare DBSCAN with K-Means and hierarchical clustering [Deep-dive]

_Key takeaways:_ (1) DBSCAN forms clusters from dense regions and labels sparse, isolated points as noise, without needing the number of clusters upfront. (2) It relies on two parameters: epsilon, the neighborhood radius, and MinPts, the minimum points for a dense region. (3) Points are classified as core (at least MinPts within epsilon), border (near a core but too few neighbors), or noise. (4) Clusters grow through density-connectivity, chains of core points each within epsilon of the next. (5) DBSCAN detects arbitrary-shaped clusters and outliers but is sensitive to parameter choice and struggles with clusters of varying density.

**Taught how (13 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Flag optional or simplified mathematical depth; Paired theory-plus-implementation with live-demo cues; Preprocessing justified by the algorithm's needs; Real-world motivation with named domains; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers; Worked example computed step by step

**Assessed by — DBSCAN (Crop Nutrient)** (EASY · Silhouette ≥ 0.35 (excl. noise) · style 4)
- Alignment: **good** — exercises DBSCAN-specific eps tuning and noise handling. Value: high. Changes: none urgent.

**Supporting content:** 1 reading, 1 in_class_quiz

---

##### DBSCAN Implementation
`unit_id: be6580b6-cf6c-4c37-a374-5864884ec76b` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: bec64ffc-c86b-418e-874a-cb85a2941110`

**Teaches:** DBSCAN, Elbow Method, Feature Scaling, Missing Values & Outlier Treatment, Standardization (Z-score Scaling)

_Flow:_ 1) Define the problem: cluster steel-plate measurements and flag unusual plates 2) Load the data and separate feature columns from label columns 3) Clean the data and scale features for distance-based clustering 4) Reduce the data to 2D with PCA for visualization 5) Run a K-Means baseline and show it cannot label noise 6) Estimate epsilon from the k-distance plot 7) Apply DBSCAN across small, medium, and large epsilon settings 8) Classify and visualize core, border, and noise points 9) Compare K-Means and DBSCAN side by side

_Outcomes (depth):_ Scale features and reduce dimensions with PCA before clustering [Applied]; Estimate epsilon from a k-distance plot [Applied]; Apply DBSCAN and read its cluster and noise outputs [Applied]; Compare DBSCAN results across different epsilon values [Deep-dive]; Classify points as core, border, or noise from a fitted model [Applied]; Contrast K-Means and DBSCAN on the same dataset [Deep-dive]

_Key takeaways:_ (1) DBSCAN is distance-based, so features must be scaled before clustering. (2) PCA reduces high-dimensional data to 2D so clusters and noise can be visualized. (3) The elbow of a k-distance plot gives a reasonable estimate for epsilon. (4) Epsilon drives the result: too small yields many noise points and fragments, while too large merges clusters and loses structure. (5) Unlike K-Means, DBSCAN labels sparse points as noise and needs no preset number of clusters, which suits anomaly detection.

**Taught how (13 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Hyperparameter sweep to reveal effect and trade-offs; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 30: Apriori Algorithm

##### Apriori Algorithm
`unit_id: 460cd724-a14f-435b-aab6-e78274072abf` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: cc46b15a-3e7e-4d14-8e90-6062da0beb12`

**Teaches:** Apriori Algorithm, Association Rule Learning

_Flow:_ 1) Motivate frequent pattern mining with a market basket example 2) Define itemsets, k-itemsets, and frequent itemsets 3) Introduce the Apriori algorithm and the Apriori pruning property 4) Define the key metrics: support, confidence, and lift 5) Explain the iterative working of Apriori 6) Form association rules with antecedent and consequent 7) Walk through a six-step market basket analysis with a worked dataset 8) Evaluate rules against thresholds and interpret business results

_Outcomes (depth):_ Compute support, confidence, and lift for a given itemset or rule [Applied]; Apply the Apriori property to prune infrequent itemsets [Applied]; Generate frequent itemsets level by level from a transaction dataset [Applied]; Derive association rules from frequent itemsets and test them against thresholds [Applied]; Explain why lift indicates the strength of an association [Explained]; Identify real-world applications of association rule mining [Introduced]

_Key takeaways:_ (1) Apriori is an unsupervised frequent-pattern-mining method that finds items which frequently appear together without any labels. (2) The Apriori property states that any superset of an infrequent itemset is also infrequent, which prunes the search space efficiently. (3) Support, confidence, and lift measure how frequent, how reliable, and how meaningful a pattern is relative to chance. (4) Frequent itemsets are built level by level and then converted into if-then association rules filtered by minimum support and confidence. (5) A lift greater than one signals an association stronger than random co-occurrence.

**Taught how (9 patterns):** Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Definition callouts for terminology; Interpret results back into the problem domain; Real-world motivation with named domains; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; When-to-use guidance with concrete triggers; Worked example computed step by step

**Assessed by — Apriori Algorithm** (EASY · rule structure + metric bounds · 5 test cases · style 3)
- Alignment: **strong** — tests actual algorithm output and the exact metrics. Value: high. Changes: none urgent.

**Supporting content:** 1 reading, 1 in_class_quiz

---

##### Apriori Implementation
`unit_id: c8a64568-31e1-4281-a51d-59360b83e43f` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: cc46b15a-3e7e-4d14-8e90-6062da0beb12`

**Teaches:** Apriori Algorithm, Association Rule Learning

_Flow:_ 1) Frame the retail market-basket problem on transaction data 2) Load and clean the raw transaction records 3) Build a transaction list and one-hot encoded basket matrix 4) Set support, confidence, and lift thresholds 5) Run Apriori to mine frequent itemsets grouped by size 6) Generate and filter association rules by confidence and lift 7) Visualize top items and strongest rules 8) Summarize rules with a language-model integration as a bonus

_Outcomes (depth):_ Preprocess raw transaction data into a one-hot basket matrix [Applied]; Mine frequent itemsets from the basket matrix [Applied]; Generate and filter association rules by confidence and lift [Applied]; Rank and interpret rules for cross-selling and product placement [Applied]; Tune support and confidence thresholds to balance pattern count against resource use [Explained]

_Key takeaways:_ (1) Transactions must be one-hot encoded into a basket matrix before Apriori can mine them. (2) Careful cleaning matters, because leaving stray count or header columns turns numbers into fake items during encoding. (3) Threshold choice trades off the number of discovered patterns against memory use and relevance. (4) A frequent-pattern library computes itemsets and association rules directly from the basket matrix. (5) Ranking rules by lift, then confidence, then support surfaces the strongest cross-sell patterns for business action.

**Taught how (10 patterns):** Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 31: Eclat

##### Eclat
`unit_id: 8a31c63c-15dc-42a9-8239-b79d30f1167f` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: ec470e1f-1a4a-4092-9037-d0e401e100b1`

**Teaches:** Association Rule Learning, Eclat Algorithm

_Flow:_ 1) Recap Apriori, association rules, and its metrics 2) Identify Apriori limitations such as repeated database scans 3) Introduce ECLAT and its vertical TID-set data format 4) Contrast horizontal breadth-first with vertical depth-first exploration 5) Compute support by intersecting transaction-ID sets 6) Walk through a worked example building 1-, 2-, and 3-itemsets 7) Prune itemsets below minimum support at each level 8) Derive recommendations and review pros, cons, and applications

_Outcomes (depth):_ Convert a transaction dataset into vertical TID-set format [Applied]; Compute itemset support by intersecting TID sets [Applied]; Generate frequent k-itemsets and prune those below minimum support [Applied]; Contrast ECLAT with Apriori across data format, search strategy, and scans [Deep-dive]; Judge when ECLAT's advantages and disadvantages apply to a dataset [Explained]

_Key takeaways:_ (1) ECLAT mines frequent itemsets using a vertical format that maps each item to the set of transaction IDs where it appears. (2) Support is simply the size of a TID set, and itemset support is found by intersecting TID sets rather than rescanning the database. (3) ECLAT explores itemsets depth-first while Apriori explores breadth-first. (4) The vertical layout makes ECLAT faster and needs fewer scans on dense data, but large TID lists can consume significant memory. (5) Itemsets grow level by level and the process stops when no candidate meets the minimum support.

**Taught how (11 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Interpret results back into the problem domain; Real-world motivation with named domains; Recap bridge from prior session; Small hand-computable toy dataset to teach mechanics; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers; Worked example computed step by step

**Supporting content:** 1 reading, 1 in_class_quiz

---

##### Eclat Implementation
`unit_id: 242bbc39-d193-4661-809d-eeec1cb7d0d1` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: ec470e1f-1a4a-4092-9037-d0e401e100b1`

**Teaches:** Association Rule Learning, Eclat Algorithm

_Flow:_ 1) Frame the grocery frequent-pattern problem 2) Clean data and group items into per-transaction baskets 3) Construct the vertical database mapping each item to its TID set 4) Prune infrequent single items by a minimum support count 5) Implement recursive ECLAT with TID-set intersections and depth-first pruning 6) Vary minimum support and analyze itemset counts by size 7) Generate association rules with support, confidence, and lift 8) Benchmark ECLAT runtime against Apriori and interpret patterns

_Outcomes (depth):_ Build a vertical TID-set database from raw transactions [Applied]; Implement a recursive ECLAT search with minimum-support pruning [Applied]; Generate association rules with support, confidence, and lift [Applied]; Analyze how minimum support affects itemset count and size [Applied]; Benchmark ECLAT runtime against an Apriori baseline [Deep-dive]

_Key takeaways:_ (1) ECLAT is implemented by building a vertical item-to-TID-set map and recursing depth-first with intersection-based support counts. (2) Early pruning, dropping candidates below minimum support before recursing deeper, keeps the search efficient. (3) Lowering the minimum support threshold rapidly increases both the number and the size of frequent itemsets. (4) Computing support as the size of intersected TID sets avoids repeated scans of the full transaction list. (5) ECLAT and Apriori recover the same frequent pairs, and TID-set intersection can run faster on this data.

**Taught how (12 patterns):** Agenda-first, Key-Takeaways-last framing; Comparison table contrasting methods; Flag optional or simplified mathematical depth; Hyperparameter sweep to reveal effect and trade-offs; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated

**Assessed by — ECLAT** (EASY · rule structure + support/confidence bounds · style 3)
- Alignment: partial — checks validity but lighter than Apriori's expected-rule match. Value: moderate. Changes: match expected rules as Apriori does.

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz, 1 mcq

---

#### Module 32: Dimentionality Reduction

##### Dimentionality Reduction
`unit_id: 1af9101f-6118-4ad9-9560-876be2c79978` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 8dfc2555-803c-4527-a19a-cf97a889a2ac`

**Teaches:** Dimensionality Reduction (PCA)

_Flow:_ 1) Challenge the intuition that more features always help 2) Introduce the curse of dimensionality and why models break in high dimensions 3) Define dimensionality reduction and its benefits 4) Distinguish feature selection from feature extraction 5) Cover feature-selection families: filter, wrapper, and embedded methods 6) Cover feature extraction and its linear versus non-linear techniques 7) Compare selection and extraction and decide when to use each

_Outcomes (depth):_ Explain how the curse of dimensionality degrades model learning [Explained]; Distinguish feature selection from feature extraction [Explained]; Categorize feature-selection techniques as filter, wrapper, or embedded [Explained]; Identify and remove irrelevant features in a worked example [Applied]; Choose between feature selection and extraction for a given dataset [Explained]

_Key takeaways:_ (1) Adding features can hurt learning because high-dimensional data becomes sparse, neighborhoods disappear, and distances lose meaning. (2) Dimensionality reduction preserves important structure while removing noise, redundancy, and irrelevant features. (3) Feature selection keeps a subset of the original features, while feature extraction creates new features by combining the originals. (4) Feature-selection methods split into filter methods, wrapper methods, and embedded methods. (5) Feature selection suits interpretable data with irrelevant features, while feature extraction suits highly correlated, high-dimensional data.

**Taught how (7 patterns):** Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Comparison table contrasting methods; Definition callouts for terminology; Myth-versus-reality misconception correction; Real-world motivation with named domains; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Dimentionality Reduction Part 2
`unit_id: d52d4bae-c9cb-403c-b2d8-18269547509a` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 797c6965-615d-455e-82e5-7bec7cab7706`

**Teaches:** Dimensionality Reduction (PCA)

_Flow:_ 1) Motivate PCA with correlated, redundant measurements 2) Introduce PCA and the idea of principal components 3) Describe components as variance-ordered and mutually perpendicular 4) Standardize the data with z-scores 5) Compute the covariance matrix 6) Perform eigen decomposition to find eigenvalues and eigenvectors 7) Select principal components by variance explained 8) Project the data onto chosen components and review PCA limitations

_Outcomes (depth):_ Standardize features with z-scores before applying PCA [Applied]; Compute a covariance matrix and solve for its eigenvalues and eigenvectors [Applied]; Select principal components based on variance explained [Applied]; Project standardized data onto a principal component to reduce dimensions [Applied]; Explain why principal components are orthogonal and variance-ordered [Explained]; Describe the limitations of PCA [Introduced]

_Key takeaways:_ (1) PCA is an unsupervised technique that builds new features called principal components as weighted combinations of the original features. (2) Principal components are ordered by the variance they capture and are perpendicular to one another so each carries new information. (3) The PCA pipeline runs standardize, covariance matrix, eigen decomposition, component selection, and projection. (4) Eigenvectors give the directions of maximum data spread and eigenvalues give the amount of variance along each direction. (5) PCA ignores class labels and is sensitive to scaling, so it can discard information useful for separating classes.

**Taught how (9 patterns):** Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Definition callouts for terminology; Paired theory-plus-implementation with live-demo cues; Preprocessing justified by the algorithm's needs; Small hand-computable toy dataset to teach mechanics; Term-by-term decomposition of a formula; When-it-fails honesty / limitations stated; Worked example computed step by step

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Dimentionality Reduction Part 3
`unit_id: c09b62c5-9b59-429b-96d3-a61b7ba17c68` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 797c6965-615d-455e-82e5-7bec7cab7706`

**Teaches:** Dimensionality Reduction (PCA)

_Flow:_ 1) Motivate LDA as separating classes rather than only spreading data 2) Introduce LDA as a supervised dimensionality reduction technique 3) Explain how LDA maximizes between-class separation and minimizes within-class spread 4) Review LDA benefits and limitations including the C-minus-one axis limit 5) Decide when to use PCA versus LDA 6) Introduce t-SNE for non-linear neighborhood-preserving visualization 7) Walk through t-SNE's three steps of similarity matching 8) Explain perplexity and its effect, then compare PCA, LDA, and t-SNE

_Outcomes (depth):_ Explain how LDA maximizes between-class separation and minimizes within-class spread [Explained]; Choose between PCA and LDA based on labels and objective [Explained]; Describe t-SNE's three-step similarity-matching procedure [Explained]; Explain the role of perplexity in shaping a t-SNE map [Explained]; Compare PCA, LDA, and t-SNE across supervision, linearity, and best use [Deep-dive]; Identify the limitations of LDA and t-SNE [Explained]

_Key takeaways:_ (1) LDA is a supervised technique that finds directions maximizing separation between classes while minimizing variation within each class. (2) LDA produces at most C-minus-one axes for C classes and assumes roughly Gaussian, balanced, linearly separable classes. (3) PCA fits label-free problems that need variance preserved, while LDA fits labeled problems whose goal is classification. (4) t-SNE is a non-linear method that preserves local neighborhoods by matching high-dimensional and low-dimensional similarity distributions through KL divergence. (5) Perplexity sets the effective number of neighbors, and t-SNE is slow, non-deterministic, and best treated only as a visualization tool.

**Taught how (8 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Hyperparameter sweep to reveal effect and trade-offs; Real-world motivation with named domains; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 in_class_quiz, 1 mcq

---

##### Dimentionality Reduction Implementation
`unit_id: f2cdf3c3-28a2-4d99-a747-7927b735b93a` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 797c6965-615d-455e-82e5-7bec7cab7706`

**Teaches:** Dimensionality Reduction (PCA), Feature Scaling, Standardization (Z-score Scaling), Train-Test Split

_Flow:_ 1) Frame the high-dimensional handwritten-digit problem 2) Load, explore, split, and standardize the pixel data 3) Train a baseline classifier on the raw pixel features 4) Apply PCA for 2D visualization and variance-based compression 5) Compare accuracy and training time before and after PCA 6) Apply supervised LDA and non-linear t-SNE projections 7) Study hyperparameter effects for PCA components and t-SNE perplexity 8) Compare PCA, LDA, and t-SNE side by side

_Outcomes (depth):_ Apply PCA to compress features by a chosen variance threshold [Applied]; Compare model accuracy and training time before and after reduction [Applied]; Project data with LDA and t-SNE and interpret the resulting plots [Applied]; Analyze the effect of PCA component count and t-SNE perplexity [Applied]; Reconstruct images from PCA components to show information loss [Applied]; Contrast PCA, LDA, and t-SNE on goal, supervision, and use [Deep-dive]

_Key takeaways:_ (1) PCA can compress hundreds of pixel features down to the components that retain most of the variance, cutting training time with little accuracy loss. (2) LDA reduces to at most C-minus-one components and groups classes, trading some accuracy for a large speed gain. (3) t-SNE produces the clearest 2D digit clusters but is slow and suited to visualization rather than training features. (4) Standardizing features and running PCA before t-SNE are practical steps that improve both speed and results. (5) Reconstructing images from more PCA components recovers more digit detail, illustrating the compression trade-off.

**Taught how (14 patterns):** Agenda-first, Key-Takeaways-last framing; Baseline model before applying the technique; Comparison table contrasting methods; Flag optional or simplified mathematical depth; Hyperparameter sweep to reveal effect and trade-offs; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Observe-and-interpret loop; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Real-world motivation with named domains; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers

**Assessed by — Dimensionality Reduction** (MEDIUM · Silhouette ≥ 0.40 on PCA-reduced · style 4)
- Alignment: **good** — PCA taught and used; full pipeline exercised. Value: high. Changes: verify PCA actually applied.

**Supporting content:** 1 reading, 1 code, 1 in_class_quiz

---

#### Module 33: Retail Customer Segmemtation

##### Problem Statement
`unit_id: 4d90fab0-e846-4792-9187-20d57e19e521` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 797c6965-615d-455e-82e5-7bec7cab7706`

**Teaches:** Data Visualization, Elbow Method, Feature Scaling, Silhouette Score, Standardization (Z-score Scaling)

_Flow:_ 1) Frame the business problem of segmenting retail customers for targeted marketing 2) Recognize that no predefined labels exist, motivating unsupervised learning 3) Understand the transaction dataset columns, scale, and two-year time span 4) Load and combine the two yearly transaction sheets and drop duplicates 5) Inspect shape, data types, missing values, and unique counts 6) Preview the three candidate clustering algorithms 7) Review the end-to-end project workflow

_Outcomes (depth):_ Frame customer segmentation as an unsupervised learning problem when no labels are available [Explained]; Describe the retail transaction dataset's columns, scale, and time span [Explained]; Load and combine two yearly transaction sheets and remove duplicate rows [Applied]; Inspect a dataset's shape, types, missing values, and unique counts [Applied]; Distinguish K-Means, Hierarchical, and DBSCAN at a conceptual level [Introduced]; Outline the end-to-end segmentation workflow from problem framing to business insights [Introduced]

_Key takeaways:_ (1) Customer segmentation is an unsupervised learning problem because no predefined customer labels exist, so groups are discovered from purchasing behavior alone. (2) Clustering fits this task because it needs no labels, handles multi-dimensional purchasing behavior, and scales to thousands of customers. (3) The dataset holds real e-commerce transactions spanning two years, with over a million line-item records covering both registered and guest customers. (4) Three candidate algorithms, K-Means, Hierarchical, and DBSCAN, each discover structure differently and are compared later in the project. (5) The project follows a defined workflow: problem framing, EDA, feature engineering, PCA, clustering, and business insights.

**Taught how (18 patterns):** Active recall via posed questions; Agenda-first, Key-Takeaways-last framing; Analogy or concrete scenario before formalism; Comparison table contrasting methods; Definition callouts for terminology; Hyperparameter sweep to reveal effect and trade-offs; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Multi-session capstone project pipeline; Observe-and-interpret loop; Preprocessing justified by the algorithm's needs; Problem statement first (business framing); Real-world motivation with named domains; Teaching code carries inline explanatory comments; Term-by-term decomposition of a formula; When-it-fails honesty / limitations stated; Worked example computed step by step

**Assessed by — Capstone — Unsupervised** (HARD · Silhouette ≥ 0.30 · style 4)
- Alignment: **good** — exercises the elbow/silhouette model-selection taught. Value: high.

**Supporting content:** 1 reading, 1 code

---

##### EDA
`unit_id: 91a9326b-bb54-4607-a062-550393cf56b3` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 797c6965-615d-455e-82e5-7bec7cab7706`

**Teaches:** Data Visualization, Exploratory Data Analysis (EDA)

_Flow:_ 1) Investigate missing Customer IDs before dropping any rows 2) Compare registered and guest transactions across quantity, price, country, and time 3) Decide to retain registered customers only 4) Remove cancelled invoices and invalid quantity or price rows 5) Engineer customer-level RFM features plus volume and diversity extras 6) Derive SpendRange and ProductDiversity features 7) Detect outliers with boxplots 8) Check feature correlations with a heatmap 9) Explore bivariate relationships between key features

_Outcomes (depth):_ Investigate missing Customer IDs before deciding whether to drop them [Applied]; Compare registered and guest transactions across quantity, price, country, and time patterns [Applied]; Clean transaction data by removing cancellations and invalid quantity or price rows [Applied]; Engineer customer-level RFM and derived behavioral features [Applied]; Interpret a correlation heatmap to detect multicollinearity among features [Explained]; Analyze bivariate relationships between engineered customer features [Applied]

_Key takeaways:_ (1) Missing Customer IDs mark guest transactions that cannot be linked across visits, so only registered customers are retained for segmentation. (2) Cancelled invoices and non-positive quantities or prices are invalid for behavioral analysis and are removed during cleaning. (3) Transaction rows are aggregated into one row per customer using RFM measures plus volume and diversity features. (4) Strong correlations among Monetary, Frequency, TotalItems, and UniqueProducts reveal multicollinearity that motivates later dimensionality reduction. (5) Spending and frequency outliers represent genuine high-value wholesale customers and are kept rather than discarded.

**Taught how (10 patterns):** Agenda-first, Key-Takeaways-last framing; Definition callouts for terminology; Domain-driven feature engineering; Investigate before dropping data; ML workflow skeleton in implementation notebooks; Multi-session capstone project pipeline; Observe-and-interpret loop; Real-world motivation with named domains; Systematic feature-by-feature / column-by-column pass; Teaching code carries inline explanatory comments

**Supporting content:** 1 reading, 1 code

---

##### Model Building -1
`unit_id: 5cefb13c-f534-41b1-bf73-d326b7b75992` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 797c6965-615d-455e-82e5-7bec7cab7706`

**Teaches:** _none tagged_

_Flow:_ 1) Preprocess features with log transformation to reduce skew 2) Scale features to a common range 3) Reduce dimensionality with PCA 4) Visualize structure in 2D with t-SNE 5) Partition customers with K-Means 6) Evaluate cluster quality with Silhouette, Dunn, and the Elbow method 7) Select the optimal K through the K-Means pipeline 8) Profile and name the resulting segments

_Outcomes (depth):_ Explain why log transformation and scaling are needed before distance-based clustering [Explained]; Describe how PCA and t-SNE reduce dimensionality for clustering and visualization [Explained]; Explain how K-Means partitions data by minimizing distance to centroids [Explained]; Identify the three metrics used to choose the optimal number of clusters [Introduced]; Outline the K-Means pipeline from metric computation to cluster profiling [Introduced]

_Key takeaways:_ (1) Log transformation reduces right-skewness so distance-based algorithms treat all customers fairly. (2) Feature scaling normalizes ranges so no single feature dominates distance calculations. (3) PCA compresses correlated features into fewer independent components, while t-SNE projects data to 2D for visual inspection. (4) K-Means partitions customers into K groups by minimizing the distance between each point and its nearest centroid. (5) The optimal number of clusters is chosen with three complementary metrics: Silhouette Score, Dunn Index, and the Elbow method.

**Taught how (8 patterns):** Agenda-first, Key-Takeaways-last framing; Comparison table contrasting methods; Hyperparameter sweep to reveal effect and trade-offs; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Multi-session capstone project pipeline; Preprocessing justified by the algorithm's needs

**Supporting content:** 1 reading, 1 code

---

##### Model Building 2
`unit_id: c9ad0d51-5fb7-4525-b9a5-7b04cf4dbc54` · `course_id: 4f4407db-75de-4209-8333-5d21abfcc064` · `topic_id: 797c6965-615d-455e-82e5-7bec7cab7706`

**Teaches:** Elbow Method, Silhouette Score

_Flow:_ 1) Build a dendrogram with Ward linkage and choose a cut level 2) Fit hierarchical clustering and profile the segments 3) Choose DBSCAN's eps from a k-distance plot and set min_samples 4) Fit DBSCAN and separate clusters from noise points 5) Experiment across eps and min_samples settings 6) Compare K-Means, Hierarchical, and DBSCAN on Silhouette and Dunn 7) Translate the chosen segments into business recommendations

_Outcomes (depth):_ Build and cut a dendrogram to choose the number of clusters [Applied]; Fit hierarchical clustering and compare it against K-Means results [Applied]; Select DBSCAN's eps from a k-distance plot and set min_samples [Applied]; Apply DBSCAN and interpret its clusters and noise points [Applied]; Compare clustering algorithms on Silhouette and Dunn while recognizing misleading scores [Deep-dive]; Translate customer segments into targeted marketing recommendations [Applied]

_Key takeaways:_ (1) Hierarchical clustering merges points into a tree, and cutting the dendrogram at the largest gap yields a natural cluster count comparable to K-Means. (2) DBSCAN groups dense regions and labels sparse points as noise, using eps and min_samples instead of a preset cluster count. (3) DBSCAN underperforms here because customer purchasing data forms a continuous gradient without the density gaps it needs. (4) A high silhouette score can mislead when one cluster absorbs nearly all points, so balanced sizes and visual separation matter alongside raw metrics. (5) The final segments split into high-value recent buyers and low-value lapsed buyers, each mapped to a distinct marketing strategy.

**Taught how (14 patterns):** Agenda-first, Key-Takeaways-last framing; Build the concept by fixing the previous version's limitation; Comparison table contrasting methods; Definition callouts for terminology; Flag optional or simplified mathematical depth; Hyperparameter sweep to reveal effect and trade-offs; Interpret results back into the problem domain; ML workflow skeleton in implementation notebooks; Metric-driven model selection, triangulated; Multi-session capstone project pipeline; Observe-and-interpret loop; Teaching code carries inline explanatory comments; When-it-fails honesty / limitations stated; When-to-use guidance with concrete triggers

**Supporting content:** 1 reading, 1 code

---

## Curriculum map (coverage / add / remove / keep)

| Concept | # units | Courses | Flag |
|---|---|---|---|
| Data Visualization | 37 | Ensemble Learning, Introdution to ML and Classification Algorithms, Supervised Learning, Unsupervised Learning | cross-course (4) |
| Train-Test Split | 31 | Ensemble Learning, Introdution to ML and Classification Algorithms, Supervised Learning, Unsupervised Learning | cross-course (4) |
| Missing Values & Outlier Treatment | 18 | Ensemble Learning, Introdution to ML and Classification Algorithms, Supervised Learning, Unsupervised Learning | cross-course (4) |
| Evaluation Metrics | 15 | Ensemble Learning, Introdution to ML and Classification Algorithms | cross-course (2) |
| Feature Scaling | 15 | Ensemble Learning, Introdution to ML and Classification Algorithms, Supervised Learning, Unsupervised Learning | cross-course (4) |
| Standardization (Z-score Scaling) | 15 | Ensemble Learning, Introdution to ML and Classification Algorithms, Supervised Learning, Unsupervised Learning | cross-course (4) |
| Regression Metrics (R2/MSE/RMSE/MAE) | 14 | Ensemble Learning, Supervised Learning | cross-course (2) |
| Label Encoding | 12 | Ensemble Learning, Introdution to ML and Classification Algorithms | cross-course (2) |
| One-Hot Encoding | 9 | Ensemble Learning, Introdution to ML and Classification Algorithms, Supervised Learning | cross-course (3) |
| HyperParameter Tuning | 7 | Introdution to ML and Classification Algorithms, Supervised Learning | cross-course (2) |
| Exploratory Data Analysis (EDA) | 6 | Ensemble Learning, Introdution to ML and Classification Algorithms, Supervised Learning, Unsupervised Learning | cross-course (4) |
| AUC-ROC Curve | 5 | Ensemble Learning |  |
| Association Rule Learning | 5 | Introdution to ML and Classification Algorithms, Unsupervised Learning | cross-course (2) |
| Class Imbalance Handling | 5 | Ensemble Learning, Introdution to ML and Classification Algorithms | cross-course (2) |
| Clustering | 5 | Introdution to ML and Classification Algorithms, Unsupervised Learning | cross-course (2) |
| Dimensionality Reduction (PCA) | 5 | Introdution to ML and Classification Algorithms, Unsupervised Learning | cross-course (2) |
| Assumptions of Linear Regression | 4 | Supervised Learning |  |
| Decision Trees | 4 | Introdution to ML and Classification Algorithms |  |
| Elbow Method | 4 | Unsupervised Learning |  |
| Ensemble Learning | 4 | Ensemble Learning |  |
| Feature Engineering | 4 | Introdution to ML and Classification Algorithms, Supervised Learning | cross-course (2) |
| K-Means Clustering | 4 | Unsupervised Learning |  |
| Silhouette Score | 4 | Unsupervised Learning |  |
| Boosting | 3 | Ensemble Learning |  |
| Gradient Boosting | 3 | Ensemble Learning |  |
| Gradient Descent | 3 | Supervised Learning |  |
| Naive Bayes | 3 | Introdution to ML and Classification Algorithms |  |
| Apriori Algorithm | 2 | Unsupervised Learning |  |
| Bagging | 2 | Ensemble Learning |  |
| Choosing K | 2 | Introdution to ML and Classification Algorithms |  |
| DBSCAN | 2 | Unsupervised Learning |  |
| Data Analysis with Pandas | 2 | Introdution to ML and Classification Algorithms |  |
| Data Cleaning | 2 | Introdution to ML and Classification Algorithms |  |
| Decision Tree Regression | 2 | Supervised Learning |  |
| Eclat Algorithm | 2 | Unsupervised Learning |  |
| Hierarchical Clustering | 2 | Unsupervised Learning |  |
| K-Nearest Neighbors (KNN) | 2 | Introdution to ML and Classification Algorithms |  |
| Logistic Regression | 2 | Supervised Learning |  |
| Multicollinearity (VIF) | 2 | Supervised Learning |  |
| Random Forest | 2 | Ensemble Learning |  |
| Regularization (L1/L2) | 2 | Supervised Learning |  |
| Simple Linear Regression | 2 | Supervised Learning |  |
| Support Vector Machines (SVM) | 2 | Introdution to ML and Classification Algorithms |  |
| Support Vector Regression | 2 | Supervised Learning |  |
| Unsupervised Learning | 2 | Introdution to ML and Classification Algorithms, Unsupervised Learning | cross-course (2) |
| AdaBoost | 1 | Ensemble Learning | thin (1 unit) |
| Bias and Variance | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Classification | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Cross-Validation | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Distance Metrics | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Euclidean Distance | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Feature Importance | 1 | Ensemble Learning | thin (1 unit) |
| Hamming Distance | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Hard vs Soft Margin | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Hyperplane and Margin | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| KNN Regression | 1 | Supervised Learning | thin (1 unit) |
| Kernel Trick | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Kernels (Linear / RBF / Polynomial) | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| ML Environment Setup | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Machine Learning | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Machine Learning Lifecycle | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Manhattan Distance | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Minkowski Distance | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Multiple Linear Regression | 1 | Supervised Learning | thin (1 unit) |
| Normalization (Min-Max Scaling) | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Polynomial Regression | 1 | Supervised Learning | thin (1 unit) |
| Probabilistic KNN | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Regression (task) | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Stacking | 1 | Ensemble Learning | thin (1 unit) |
| Supervised Learning | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Support Vectors | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| Voting | 1 | Ensemble Learning | thin (1 unit) |
| Weighted KNN | 1 | Introdution to ML and Classification Algorithms | thin (1 unit) |
| XGBoost | 1 | Ensemble Learning | thin (1 unit) |

Gap: **Entropy / Information Gain** is taught (Decision Tree) but not in the concept vocabulary — candidate to ADD. Thin concepts are reinforce-or-keep; cross-course concepts are dedupe-or-keep (spiral curriculum).

## Two-tier assessment

Assessment runs at two levels (matching the pattern across the NxtWave KBs):

- **Formative (per session):** 55 MCQ-practice sets and 65 in-class quizzes attached to individual sessions, plus 93 readings.
- **Summative (per module):** 7 module quizzes closing the modules.

## Coding assignment analysis (program-wide)

The program-wide assignment analysis: coverage, evaluation model, the six evaluation styles, QC issues, design patterns, and curriculum-level findings. The per-assignment records are woven into each unit above under **Assessed by** (not repeated here).

Analysis of the program's IDE coding assignments: what each session intended to teach vs what the
assignment actually tests, how much it helps students, changes worth making. Self-contained; no database needed to read it.

Built from three inputs: the `Classical ML coding assignments` sheet (session → assignment →
resource mapping), the assignment content pulled from the **canonical Drive set** (problem
statement, `test_solution.py`, `question.json` toughness/weightage), and the session knowledge base
(concepts + learning outcomes + depth). The evaluation mechanics follow the IDE pytest model
(structure tests + a hidden-ground-truth performance test).

### Coverage

27 rows in the sheet. **26 analyzed** (below), pulled from the canonical Drive folder
`Classical ML/.../Coding Assignments` (22-folder set + the 4 in the older "Coding Assignments"
folder: SVM, Simple LR, Gradient Descent, Regularisation). **"Evaluation Metrics" is the only one
not analyzed — it has no resource in the sheet, so no assignment exists for it yet.**

### How these assignments are graded (the evaluation model)

The dominant contract is a **Kaggle-style submission** checked against a hidden ground truth, with
a near-identical 5-test skeleton:

| Test | Checks | Typical weight |
|---|---|---|
| PY1 | Notebook runs without error | 5–20 |
| PY2 | Output variable exists / required estimator present / correct columns | 15–40 |
| PY3 | Correct row count (one prediction per test sample) | 15–30 |
| PY4 | Values are valid (label set / numeric / finite / range) | 15–40 |
| PY5(+) | **Performance threshold** (accuracy / R² / MSE / RMSE / Silhouette / precision) | 25–50 |

Detection is by **variable name** (`submission_df`, `submission`, `cluster_labels`) or by
**estimator type** (`isinstance` against the sklearn class). Thresholds are set ~10–20% below a
reference solution.

### Six evaluation styles (this is the key lens)

Not all assignments test the same way. Sorting them by *what the tests actually enforce* is the
most useful view — it tells you which assignments reinforce the session's concept and which only
check plumbing:

1. **Verifies the exact technique** — asserts the intended estimator is present/trained:
   Gradient Descent (`isinstance SGDRegressor` + StandardScaler used), Regularisation (Ridge *and*
   Lasso present), Simple LR (`LinearRegression` present), Capstone-Classification (a fitted
   classifier + an EDA plot must exist). **Best conceptual alignment.**
2. **Forces multiple techniques via required output columns** — KNN_SVM_DT (KNN/SVM/Decision Tree
   columns), Bagging (Bagged-DT + RF columns), Voting & Stacking (voting_pred + stacking_pred).
   The student must actually build each named method.
3. **Tests real algorithm output values** — EDA Project (15 specific aggregations), Apriori
   (frozenset rules + support/confidence/lift bounds + expected-rule match), ECLAT (rule bounds).
4. **Clustering: Silhouette threshold + required tuning** — K-Means, Hierarchical, DBSCAN,
   Dimensionality Reduction, Unsupervised Capstone. Each forces the model-selection step the
   session teaches (K sweep / linkage choice / eps tuning / PCA).
5. **Meaningful single-metric threshold** — Multiple LR (R²≥0.90), Polynomial (MSE<17 & R²>0.90),
   Boosting (80%), Decision Tree (85%), the HARD capstones (RMSE/accuracy on real data).
6. **Technique-agnostic + lenient** — scored purely on a formatted submission clearing a soft bar,
   with no check that the taught method was used: **KNN (65%), Naive Bayes (75%), Logistic
   Regression (65%), Hyperparameter Tuning (65%)**. A default or even wrong-algorithm model passes.
   **This is where intent and test diverge most.**

### QC issues found (worth fixing at source)

- **KNN_SVM_DT** — `assert svm_mse > 1` gates nothing (MSE > 1 is a trivial lower bound; it should
  be an upper bound like `< N`). The SVM third of the assignment is effectively ungraded.
- **SVM** — the student-facing text says accuracy ≥ 0.82 but the test asserts ≥ 0.80 (told a higher
  bar than graded). Harmless to pass rates, but inconsistent.
- **Lenient thresholds** — KNN 65% and Logistic 65% on easy datasets, K-Means Silhouette ≥ 0.25.
  A reference solution clears these by a wide margin, so PY5 rarely bites.
- **Technique not enforced** in the six style-6 assignments above (see the recommendation below).

### Assignment design patterns (what recurs)

1. **Submission-file contract** — a named output + hidden ground truth + threshold. Universal.
2. **5-test skeleton** — runs → structure/estimator → row count → valid values → performance.
3. **Real-world named domains** — apple quality, heart stroke, phishing, Titanic, trip duration,
   earthquake damage, crop nutrients, vitamin deficiency. Matches the sessions' motivation style.
4. **Front-loaded plumbing** in style-6 (structure ~70% / performance ~30%); style-1/5 push more
   weight onto the estimator/performance checks.
5. **Threshold ~10–20% below reference** — per the IDE authoring skill.

---

### Curriculum-level findings

1. **The gap is concentrated, not universal.** Only the **single-model classification** assignments
   (KNN, Naive Bayes, Logistic Regression, Hyperparameter Tuning — plus Boosting/Decision Tree
   partially) are technique-agnostic and lenient. The regression-mechanics (Gradient Descent,
   Regularisation, Simple LR), multi-technique (KNN_SVM_DT, Bagging, Voting & Stacking), clustering,
   and algorithm-output (EDA, Apriori) assignments **are well aligned** — several verify the exact
   estimator or force the taught method structurally. The set is stronger than a first glance suggests.
2. **Highest-leverage single fix:** add a light "intended-technique present" check to the ~4
   technique-agnostic classification assignments — the pattern already exists in the good ones
   (`isinstance(model, SGDRegressor)`, `ridge_model is not None`), so it's a copy-paste, not new design.
3. **Fix the two concrete bugs:** KNN_SVM_DT `svm_mse > 1` (gates nothing) and the SVM 0.82-vs-0.80
   display/assert mismatch.
4. **Calibrate soft thresholds** against a reference solution (KNN/Logistic 65%, K-Means 0.25).
5. **Normalize toughness labels** (Bagging has none; Decision Tree at 85% is "EASY" while a 70%
   3-class capstone is "HARD" — labels track scope, not the score bar).
6. **KNN_SVM_DT spans three sessions' content** (KNN + SVM + DT regression) but is mapped to one
   session — worth noting for portal linkage.

### Provenance

Assignment content pulled from the canonical Google Drive set (`Coding Assignments` under Classical
ML, owner dsml.contentteam, 2026-06-08 batch) via the read-only Drive client. Session intent from
the Classical ML knowledge base. This file is self-contained; regenerate by re-pulling the Drive
folders and re-reading the KB.

## Pedagogy patterns (full)

The complete pattern analysis: each pattern's description, rules, the sessions it was observed in, evidence, and the supporting content available in those sessions.

Descriptive teaching patterns extracted from the course's **slides + notebooks** (text only; no visual/slide-image pass yet). Each pattern lists the sessions it was observed in, with evidence, and the supporting content available in those sessions. Patterns are DESCRIPTIVE (what the content does), not prescriptive mandates.

**40 patterns**, **878 observations** across **95 sessions**. By category: 20 ML-native patterns, 3 Session-flow patterns, 2 Learning strategies, 6 Core principles, 5 Cross-cutting patterns, 4 Notebook / implementation patterns.

> Not captured (needs a slide-image pass): visual teaching mechanics — progressive reveals, diagram-first ordering, color idioms, layout. See the project notes.

### ML-native patterns

#### Real-world motivation with named domains  `ml`
Topics are anchored in specific, named application domains rather than abstract data, and the domain is carried through the examples.

Rules:
- Anchor the concept in a concrete named domain (spam, fraud, churn, medical, retail) from the outset.
- Reuse that same domain across the session's examples for continuity.

Observed in 55 session(s): AUC - ROC Curve (Ensemble Learning), Apriori Algorithm (Unsupervised Learning), Bagging (Ensemble Learning), Bias and Variance (Introdution to ML and Classification Algorithms), Boosting Implementation - Classification (Ensemble Learning), Boosting Implementation - Regression (Ensemble Learning), Boosting methods and Adaboost (Ensemble Learning), Build Base ML model (Supervised Learning), Course OverView (Introdution to ML and Classification Algorithms), DA with Pandas Part - 1 (Introdution to ML and Classification Algorithms), DA with Pandas Part - 2 (Introdution to ML and Classification Algorithms), DBSCAN (Unsupervised Learning), Decision Tree Implementation Part 1 (Introdution to ML and Classification Algorithms), Decision Tree Implementation Part 2 (Introdution to ML and Classification Algorithms), Dimentionality Reduction (Unsupervised Learning), Dimentionality Reduction Implementation (Unsupervised Learning), Dimentionality Reduction Part 3 (Unsupervised Learning), EDA (Ensemble Learning), EDA (Introdution to ML and Classification Algorithms), EDA (Supervised Learning), EDA (Unsupervised Learning), Eclat (Unsupervised Learning), Evaluation Metrics (Introdution to ML and Classification Algorithms), Evaluation Metrics Implementation (Introdution to ML and Classification Algorithms), Gradient Boosting Classification (Ensemble Learning), Gradient Boosting Regression (Ensemble Learning), HyperParameter Tuning Part2 (Introdution to ML and Classification Algorithms), Introduction to Machine Learning (Introdution to ML and Classification Algorithms), Introduction to Unsupervised Learning (Unsupervised Learning), K-Means Clustering (Unsupervised Learning), K-Means implementation (Unsupervised Learning), KNN Implementation with Scikit-Learn (Introdution to ML and Classification Algorithms), KNN Regression (Supervised Learning), Logistic Regression (Supervised Learning), Missing Values and Outliers Treatment (Supervised Learning), Model Building 1 (Ensemble Learning), Model Building 2 (Ensemble Learning), Multiple Linear Regression (Supervised Learning), Naive Bayes Implementation (Introdution to ML and Classification Algorithms), Naive Bayes Part 1 (Introdution to ML and Classification Algorithms), Naive Bayes Part 2 (Introdution to ML and Classification Algorithms), Polynomial Regression (Supervised Learning), Problem Statement (Ensemble Learning), Problem Statement (Introdution to ML and Classification Algorithms), Problem Statement (Supervised Learning), Problem Statement (Unsupervised Learning), Random Forest (Ensemble Learning), Random Forest Implementation (Ensemble Learning), SVM (Introdution to ML and Classification Algorithms), SVM Implementation (Introdution to ML and Classification Algorithms), Simple Linear Regression (Supervised Learning), Support Vector Regression (Supervised Learning), Voting (Ensemble Learning), XG boost (Ensemble Learning)

Evidence:
- SVM Implementation (Introdution to ML and Classification Algorithms): Anchored in mushroom classification with named anatomy features: cap, scales, gills, annulus (ring), stipe (stem), volva.
- K-Means implementation (Unsupervised Learning): Mall customer segmentation for 'personalized offers and loyalty programs' carried through the whole notebook.
- Evaluation Metrics (Introdution to ML and Classification Algorithms): Email spam filtering carried throughout; cancer detection used for minimizing false negatives; phishing for recall.
- AUC - ROC Curve (Ensemble Learning): Spam detection anchors TP/FN/FP/TN (spam vs legitimate emails); high TPR tied to medical diagnosis, fraud detection, cancer screening.

Supporting content in these sessions: 53 reading, 40 in_class_quiz, 30 mcq

---

#### ML workflow skeleton in implementation notebooks  `notebook`
Implementation notebooks follow a consistent load -> inspect/EDA -> clean/preprocess -> train -> evaluate pipeline, sometimes mirrored on a slide.

Rules:
- Order the notebook as load data -> inspect/EDA -> clean & preprocess -> train model -> evaluate.
- Use the same section headings across notebooks so the pipeline is recognizable.
- Where possible, show the pipeline as an ordered diagram on a slide.

Observed in 53 session(s): Apriori Implementation (Unsupervised Learning), Assumptions of Linear Regression Implementation (Supervised Learning), Bagging (Ensemble Learning), Boosting Implementation - Classification (Ensemble Learning), Boosting Implementation - Regression (Ensemble Learning), Build Base ML model (Introdution to ML and Classification Algorithms), Build Base ML model (Supervised Learning), Build Final ML model (Supervised Learning), Build ML model and Conclusion (Introdution to ML and Classification Algorithms), DA with Pandas Part - 1 (Introdution to ML and Classification Algorithms), DA with Pandas Part - 2 (Introdution to ML and Classification Algorithms), DBSCAN Implementation (Unsupervised Learning), Decision Tree Implementation Part 1 (Introdution to ML and Classification Algorithms), Decision Tree Regression Implementation (Supervised Learning), Dimentionality Reduction Implementation (Unsupervised Learning), EDA (Ensemble Learning), EDA (Introdution to ML and Classification Algorithms), EDA (Supervised Learning), EDA (Unsupervised Learning), Eclat Implementation (Unsupervised Learning), Evaluation Metrics Implementation (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Supervised Learning), Feature Engineering - 2 (Introdution to ML and Classification Algorithms), Feature Engineering - 2 (Supervised Learning), Feature Importance Techniques (Ensemble Learning), Gradient Descent (Supervised Learning), Hierarchical Clustering Implementation (Unsupervised Learning), HyperParameter Tuning Part1 (Introdution to ML and Classification Algorithms), HyperParameter Tuning Part2 (Introdution to ML and Classification Algorithms), K-Means implementation (Unsupervised Learning), KNN Implementation with Scikit-Learn (Introdution to ML and Classification Algorithms), KNN Regression (Supervised Learning), Logistic Regression Implementation (Supervised Learning), Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms), Missing Values and Outliers Treatment (Supervised Learning), Model Building -1 (Unsupervised Learning), Model Building 1 (Ensemble Learning), Model Building 2 (Ensemble Learning), Model Building 2 (Unsupervised Learning), Multiple Linear Regression (Supervised Learning), Naive Bayes Implementation (Introdution to ML and Classification Algorithms), Problem Statement (Ensemble Learning), Problem Statement (Supervised Learning), Problem Statement (Unsupervised Learning), Random Forest Implementation (Ensemble Learning), Regularization Implementation (Supervised Learning), SLR Implementation (Supervised Learning), SVM (Introdution to ML and Classification Algorithms), SVM Implementation (Introdution to ML and Classification Algorithms), Stacking (Ensemble Learning), Support Vector Regression Implementation (Supervised Learning), Voting (Ensemble Learning)

Evidence:
- SVM Implementation (Introdution to ML and Classification Algorithms): Workflow diagram slide: Define Problem -> Load data -> Perform EDA -> Train Test Split -> Build SVM Model -> Evaluate Model.
- K-Means implementation (Unsupervised Learning): Import -> Load -> Data Inspection -> Data Cleaning -> Select features -> Scaling -> Model Building -> Elbow -> Silhouette -> Dunn -> Interpret; slide mirrors 'Define Problem -> Load -> Clean -> Featur
- Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms): Notebook continues the pipeline: Load -> EDA -> 'Missing value Imputation' -> 'Outlier Treatment'.
- DA with Pandas Part - 1 (Introdution to ML and Classification Algorithms): Notebook runs a consistent load -> info/inspect -> drop cols -> rename -> dropna -> clean columns -> drop duplicates -> save -> visualize pipeline, mirrored on the Agenda (Problem Statement -> Data Cl

Supporting content in these sessions: 53 reading, 28 in_class_quiz, 25 mcq

---

#### When-it-fails honesty / limitations stated  `ml`
Where a method breaks or misleads is surfaced explicitly, and methods are paired with their failure modes or a pros-and-cons view.

Rules:
- State the failure modes and assumptions where the method breaks, alongside its strengths.
- Where a metric or method can mislead, show the misleading case concretely.

Observed in 49 session(s): AUC - ROC Curve (Ensemble Learning), Apriori Implementation (Unsupervised Learning), Assumptions of Linear Regression Implementation (Supervised Learning), Bagging (Ensemble Learning), Bias and Variance (Introdution to ML and Classification Algorithms), Boosting methods and Adaboost (Ensemble Learning), DBSCAN (Unsupervised Learning), DBSCAN Implementation (Unsupervised Learning), Decision Tree Implementation Part 2 (Introdution to ML and Classification Algorithms), Decision Tree Regression Implementation (Supervised Learning), Dimentionality Reduction Implementation (Unsupervised Learning), Dimentionality Reduction Part 2 (Unsupervised Learning), Dimentionality Reduction Part 3 (Unsupervised Learning), EDA (Ensemble Learning), Eclat (Unsupervised Learning), Eclat Implementation (Unsupervised Learning), Evaluation Metrics (Introdution to ML and Classification Algorithms), Evaluation Metrics Implementation (Introdution to ML and Classification Algorithms), Feature Engineering - 2 (Introdution to ML and Classification Algorithms), Feature Importance Techniques (Ensemble Learning), Gradient Boosting Classification (Ensemble Learning), Gradient Boosting Regression (Ensemble Learning), Gradient Descent (Supervised Learning), Gradient Descent Part - 2 (Supervised Learning), Hierarchical Clustering (Unsupervised Learning), Hierarchical Clustering Implementation (Unsupervised Learning), HyperParameter Tuning Part1 (Introdution to ML and Classification Algorithms), K-Means Clustering Part 2 (Unsupervised Learning), K-Means Clustering Part 3 (Unsupervised Learning), K-Means implementation (Unsupervised Learning), KNN Implementation with Scikit-Learn (Introdution to ML and Classification Algorithms), KNN Regression (Supervised Learning), Logistic Regression Implementation (Supervised Learning), Model Building 1 (Ensemble Learning), Model Building 2 (Ensemble Learning), Model Building 2 (Unsupervised Learning), Naive Bayes Implementation (Introdution to ML and Classification Algorithms), Naive Bayes Part 1 (Introdution to ML and Classification Algorithms), Polynomial Regression (Supervised Learning), Problem Statement (Unsupervised Learning), Random Forest Implementation (Ensemble Learning), Regularization Implementation (Supervised Learning), SLR Implementation (Supervised Learning), SVM (Introdution to ML and Classification Algorithms), SVM Implementation (Introdution to ML and Classification Algorithms), Stacking (Ensemble Learning), Support Vector Regression Implementation (Supervised Learning), Voting (Ensemble Learning), XG boost (Ensemble Learning)

Evidence:
- SVM Implementation (Introdution to ML and Classification Algorithms): Explicit Cons slide: Computationally Intensive, Sensitive to Noise, Difficult to Interpret, Requires Feature Scaling.
- K-Means implementation (Unsupervised Learning): Slide cons: 'Difficulty Choosing K; Sensitive to Initial Values; Order-Dependent Results; Sensitive to Rescaling; ... it struggles with complex or irregular cluster shapes because it assumes clusters 
- Evaluation Metrics (Introdution to ML and Classification Algorithms): 'The accuracy paradox occurs when a model shows high accuracy but fails to effectively identify one class'; 'can give a false sense of security'.
- AUC - ROC Curve (Ensemble Learning): 'Limitations' slides: on highly imbalanced data a model with many false positives still gets small FPR, so 'the ROC curve may look strong and the AUC may be high, even though the model is not actually

Supporting content in these sessions: 48 reading, 39 in_class_quiz, 34 mcq

---

#### Observe-and-interpret loop  `notebook`
Every plot or computed output is followed by a short written interpretation phrased as an observation or decision, not just code narration.

Rules:
- After each plot or output, write a one-to-three sentence interpretation of what it shows.
- Phrase interpretations as observations and decisions, converting numbers into meaning.

Observed in 46 session(s): Apriori Implementation (Unsupervised Learning), Assumptions of Linear Regression Implementation (Supervised Learning), Bagging (Ensemble Learning), Boosting Implementation - Classification (Ensemble Learning), Boosting Implementation - Regression (Ensemble Learning), Build Base ML model (Introdution to ML and Classification Algorithms), Build Base ML model (Supervised Learning), Build Final ML model (Supervised Learning), Build ML model and Conclusion (Introdution to ML and Classification Algorithms), DA with Pandas Part - 1 (Introdution to ML and Classification Algorithms), DA with Pandas Part - 2 (Introdution to ML and Classification Algorithms), DBSCAN Implementation (Unsupervised Learning), Dimentionality Reduction Implementation (Unsupervised Learning), EDA (Ensemble Learning), EDA (Introdution to ML and Classification Algorithms), EDA (Supervised Learning), EDA (Unsupervised Learning), Eclat Implementation (Unsupervised Learning), Evaluation Metrics (Introdution to ML and Classification Algorithms), Evaluation Metrics Implementation (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Supervised Learning), Feature Engineering - 2 (Introdution to ML and Classification Algorithms), Feature Engineering - 2 (Supervised Learning), Feature Importance Techniques (Ensemble Learning), Gradient Descent (Supervised Learning), Hierarchical Clustering Implementation (Unsupervised Learning), HyperParameter Tuning Part1 (Introdution to ML and Classification Algorithms), HyperParameter Tuning Part2 (Introdution to ML and Classification Algorithms), K-Means implementation (Unsupervised Learning), KNN Implementation with Scikit-Learn (Introdution to ML and Classification Algorithms), Logistic Regression Implementation (Supervised Learning), Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms), Missing Values and Outliers Treatment (Supervised Learning), Model Building 1 (Ensemble Learning), Model Building 2 (Ensemble Learning), Model Building 2 (Unsupervised Learning), Multiple Linear Regression (Supervised Learning), Naive Bayes Implementation (Introdution to ML and Classification Algorithms), Problem Statement (Unsupervised Learning), Random Forest Implementation (Ensemble Learning), Regularization Implementation (Supervised Learning), SLR Implementation (Supervised Learning), SVM (Introdution to ML and Classification Algorithms), Stacking (Ensemble Learning), Voting (Ensemble Learning)

Evidence:
- K-Means implementation (Unsupervised Learning): '### Observations (Elbow Graph): WCSS decreases sharply from K=1 to K=5... Hence, K=5 is a good choice'; '### Observations - Silhouette... Decision: K = 5.'
- Evaluation Metrics (Introdution to ML and Classification Algorithms): Each computed metric is interpreted: 'Accuracy 98%... can be misleading', 'correct 67% of the time... significant false positives', 'only identifies 40%... many relevant emails missed'.
- Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms): 'It can be seen that in loan amount term variable, the value of 360 is repeating the most. So we will replace the missing values... using the mode'; after log transform: 'Now the distribution looks mu
- DA with Pandas Part - 1 (Introdution to ML and Classification Algorithms): Every plot followed by a written interpretation, e.g. 'By observing the graph, we can say that 25000+ restaurants are accepting online order' and 'levelle Road is a best location by comparing votes an

Supporting content in these sessions: 46 reading, 23 in_class_quiz, 20 mcq

---

#### Comparison table contrasting methods  `ml`
A new algorithm is contrasted against a sibling or alternative across shared dimensions, using the contrast to clarify when each applies.

Rules:
- Place the new method beside an alternative in a side-by-side table over shared dimensions (assumptions, speed, output).
- Use the contrast to state when to prefer each method.

Observed in 43 session(s): AUC - ROC Curve (Ensemble Learning), Bagging (Ensemble Learning), Boosting Implementation - Classification (Ensemble Learning), Boosting Implementation - Regression (Ensemble Learning), Boosting methods and Adaboost (Ensemble Learning), DBSCAN (Unsupervised Learning), DBSCAN Implementation (Unsupervised Learning), Decision Tree Part 2 (Introdution to ML and Classification Algorithms), Dimentionality Reduction (Unsupervised Learning), Dimentionality Reduction Implementation (Unsupervised Learning), Dimentionality Reduction Part 3 (Unsupervised Learning), Eclat (Unsupervised Learning), Eclat Implementation (Unsupervised Learning), Evaluation Metrics Implementation (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Introdution to ML and Classification Algorithms), Feature Engineering - 2 (Supervised Learning), Feature Importance Techniques (Ensemble Learning), Gradient Boosting Regression (Ensemble Learning), Gradient Descent Part - 1 (Supervised Learning), Gradient Descent Part - 2 (Supervised Learning), Hierarchical Clustering (Unsupervised Learning), Hierarchical Clustering Implementation (Unsupervised Learning), HyperParameter Tuning Part1 (Introdution to ML and Classification Algorithms), HyperParameter Tuning Part2 (Introdution to ML and Classification Algorithms), Introduction to Ensemble Algorithms (Ensemble Learning), Introduction to Machine Learning (Introdution to ML and Classification Algorithms), Introduction to Unsupervised Learning (Unsupervised Learning), KNN (Introdution to ML and Classification Algorithms), KNN Advanced (Introdution to ML and Classification Algorithms), Logistic Regression (Supervised Learning), Model Building -1 (Unsupervised Learning), Model Building 1 (Ensemble Learning), Model Building 2 (Ensemble Learning), Model Building 2 (Unsupervised Learning), Naive Bayes Implementation (Introdution to ML and Classification Algorithms), Naive Bayes Part 1 (Introdution to ML and Classification Algorithms), Problem Statement (Unsupervised Learning), Random Forest (Ensemble Learning), Regularization (Supervised Learning), SVM (Introdution to ML and Classification Algorithms), Simple Linear Regression (Supervised Learning), Stacking (Ensemble Learning), Voting (Ensemble Learning)

Evidence:
- AUC - ROC Curve (Ensemble Learning): 'ROC Curve & PR Curve' slide compares the same two classifiers on both plots; PR reveals blue keeps higher precision while green drops, a difference ROC hid.
- Hierarchical Clustering (Unsupervised Learning): Linkage table (Single/Complete/Average/Ward x Best For/Cluster Shape/Noise Sensitivity/Distance) and a 'Hierarchical Clustering vs K-Means' table across Approach/Number of Clusters/Output/Flexibility/
- Dimentionality Reduction (Unsupervised Learning): 'Feature Selection vs Feature Extraction' table: selects existing vs creates new, drops irrelevant vs combines, meaning preserved vs meaning may change.
- Feature Importance Techniques (Ensemble Learning): Table contrasts RFC vs LOFO by Approach (impurity reduction vs retrain leaving out features) and Advantages (fast/intuitive vs model-agnostic/direct impact).

Supporting content in these sessions: 42 reading, 36 in_class_quiz, 27 mcq

---

#### Problem statement first (business framing)  `notebook`
Implementation notebooks open with a plain-language problem statement naming the goal and the business use, restated as the modeling objective.

Rules:
- Begin the notebook with a problem statement in plain language naming the goal and its business context.
- Restate the goal as the concrete modeling objective before coding.

Observed in 43 session(s): Apriori Implementation (Unsupervised Learning), Assumptions of Linear Regression Implementation (Supervised Learning), Bagging (Ensemble Learning), Boosting Implementation - Classification (Ensemble Learning), Boosting Implementation - Regression (Ensemble Learning), Build Base ML model (Supervised Learning), Build Final ML model (Supervised Learning), Build ML model and Conclusion (Introdution to ML and Classification Algorithms), DA with Pandas Part - 1 (Introdution to ML and Classification Algorithms), DA with Pandas Part - 2 (Introdution to ML and Classification Algorithms), DBSCAN Implementation (Unsupervised Learning), Decision Tree Implementation Part 1 (Introdution to ML and Classification Algorithms), Decision Tree Implementation Part 2 (Introdution to ML and Classification Algorithms), Decision Tree Regression Implementation (Supervised Learning), Dimentionality Reduction Implementation (Unsupervised Learning), EDA (Supervised Learning), Eclat Implementation (Unsupervised Learning), Evaluation Metrics Implementation (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Supervised Learning), Feature Engineering - 2 (Supervised Learning), Gradient Descent (Supervised Learning), Hierarchical Clustering Implementation (Unsupervised Learning), HyperParameter Tuning Part2 (Introdution to ML and Classification Algorithms), K-Means implementation (Unsupervised Learning), KNN Implementation with Scikit-Learn (Introdution to ML and Classification Algorithms), KNN Regression (Supervised Learning), Logistic Regression Implementation (Supervised Learning), Missing Values and Outliers Treatment (Supervised Learning), Model Building 1 (Ensemble Learning), Model Building 2 (Ensemble Learning), Multiple Linear Regression (Supervised Learning), Naive Bayes Implementation (Introdution to ML and Classification Algorithms), Problem Statement (Ensemble Learning), Problem Statement (Introdution to ML and Classification Algorithms), Problem Statement (Supervised Learning), Problem Statement (Unsupervised Learning), Random Forest Implementation (Ensemble Learning), Regularization Implementation (Supervised Learning), SLR Implementation (Supervised Learning), SVM (Introdution to ML and Classification Algorithms), Stacking (Ensemble Learning), Support Vector Regression Implementation (Supervised Learning), Voting (Ensemble Learning)

Evidence:
- K-Means implementation (Unsupervised Learning): Notebook opens '### Problem Statement: The mall aims to segment its customers to enable targeted marketing strategies... Build an unsupervised K-Means clustering to group customers into K distinct seg
- DA with Pandas Part - 1 (Introdution to ML and Classification Algorithms): Slide 'Problem Statement: Data Analysis and Visualization of Restaurant Trends' with an Objective on the Zomato dataset; notebook opens with '# About Dataset' describing Bangalore restaurants.
- DA with Pandas Part - 2 (Introdution to ML and Classification Algorithms): Slide 'Problem Statement: Data Analysis and Visualization of Restaurant Trends' + notebook '# About Dataset' on Zomato Bangalore restaurants.
- Build ML model and Conclusion (Introdution to ML and Classification Algorithms): Opens as '# Classification workflow : Loan Approval Project', naming the classification goal before any code.

Supporting content in these sessions: 43 reading, 26 in_class_quiz, 22 mcq

---

#### When-to-use guidance with concrete triggers  `ml`
A method is accompanied by a when-to-use segment giving concrete situational triggers and example domains for reaching for it.

Rules:
- Include a 'when to use' segment with concrete triggers for choosing the method.
- Attach an example domain to each trigger so the guidance is actionable.

Observed in 35 session(s): AUC - ROC Curve (Ensemble Learning), Apriori Algorithm (Unsupervised Learning), Bagging (Ensemble Learning), Boosting Implementation - Regression (Ensemble Learning), DBSCAN (Unsupervised Learning), Dimentionality Reduction (Unsupervised Learning), Dimentionality Reduction Implementation (Unsupervised Learning), Dimentionality Reduction Part 3 (Unsupervised Learning), EDA (Ensemble Learning), EDA (Introdution to ML and Classification Algorithms), Eclat (Unsupervised Learning), Evaluation Metrics (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Supervised Learning), Feature Engineering - 2 (Supervised Learning), Feature Importance Techniques (Ensemble Learning), Gradient Boosting Classification (Ensemble Learning), Gradient Descent Part - 1 (Supervised Learning), Gradient Descent Part - 2 (Supervised Learning), Hierarchical Clustering (Unsupervised Learning), HyperParameter Tuning Part1 (Introdution to ML and Classification Algorithms), KNN (Introdution to ML and Classification Algorithms), KNN Advanced (Introdution to ML and Classification Algorithms), Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms), Model Building 2 (Ensemble Learning), Model Building 2 (Unsupervised Learning), Naive Bayes Part 2 (Introdution to ML and Classification Algorithms), Polynomial Regression (Supervised Learning), Problem Statement (Ensemble Learning), Random Forest Implementation (Ensemble Learning), Regularization (Supervised Learning), SVM (Introdution to ML and Classification Algorithms), Stacking (Ensemble Learning), XG boost (Ensemble Learning)

Evidence:
- Evaluation Metrics (Introdution to ML and Classification Algorithms): 'Use precision when false positives can lead to significant problems (misclassifying client emails)'; 'Use recall when it is crucial to catch every spam (phishing)'.
- Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms): Outlier Treatment Methods give conditional triggers: 'Remove: If the outliers are errors or irrelevant', 'Cap or Floor: Replace outliers with the nearest value within an acceptable range', 'Transforma
- AUC - ROC Curve (Ensemble Learning): 'High TPR is important in domains like medical diagnosis, fraud detection, cancer screening'; 'PR Curve is well suited for imbalanced datasets.'
- Hierarchical Clustering (Unsupervised Learning): Linkage table 'Best For' column: Single->Connectivity, Complete->Tight clusters, Average->General use, Ward->Variance minimization; 'we use any one of them, depending on context and domain of the prob

Supporting content in these sessions: 35 reading, 25 in_class_quiz, 17 mcq

---

#### Metric-driven model selection, triangulated  `ml`
Hyperparameters/models are chosen by explicit metrics rather than by fiat, and when one metric is inconclusive it is corroborated by others and by interpretability.

Rules:
- Choose K/model/threshold using a stated metric, not an arbitrary pick.
- When one metric is inconclusive, corroborate with additional metrics and interpretability, and say so explicitly.

Observed in 32 session(s): Apriori Implementation (Unsupervised Learning), Bagging (Ensemble Learning), Boosting Implementation - Classification (Ensemble Learning), Boosting Implementation - Regression (Ensemble Learning), Build Base ML model (Introdution to ML and Classification Algorithms), Build Base ML model (Supervised Learning), Build Final ML model (Supervised Learning), Build ML model and Conclusion (Introdution to ML and Classification Algorithms), DBSCAN Implementation (Unsupervised Learning), Decision Tree Regression Implementation (Supervised Learning), Dimentionality Reduction Implementation (Unsupervised Learning), Eclat Implementation (Unsupervised Learning), Evaluation Metrics Implementation (Introdution to ML and Classification Algorithms), Feature Engineering - 2 (Supervised Learning), Feature Importance Techniques (Ensemble Learning), HyperParameter Tuning Part1 (Introdution to ML and Classification Algorithms), HyperParameter Tuning Part2 (Introdution to ML and Classification Algorithms), K-Means Clustering Part 2 (Unsupervised Learning), K-Means Clustering Part 3 (Unsupervised Learning), K-Means implementation (Unsupervised Learning), KNN Implementation with Scikit-Learn (Introdution to ML and Classification Algorithms), KNN Regression (Supervised Learning), Model Building -1 (Unsupervised Learning), Model Building 1 (Ensemble Learning), Model Building 2 (Ensemble Learning), Model Building 2 (Unsupervised Learning), Problem Statement (Unsupervised Learning), Random Forest Implementation (Ensemble Learning), Regularization Implementation (Supervised Learning), Stacking (Ensemble Learning), Support Vector Regression Implementation (Supervised Learning), Voting (Ensemble Learning)

Evidence:
- K-Means implementation (Unsupervised Learning): 'Since there is no clear peak, the Dunn Index alone is not sufficient'; 'K = 5 is chosen by balancing the Silhouette Score, Dunn Index, and interpretability.'
- K-Means Clustering Part 2 (Unsupervised Learning): K chosen by WCSS/Elbow, corroborated by Silhouette when the elbow is ambiguous; 'k = 2 has the highest score (0.82) -> Optimal k = 2.'
- Feature Importance Techniques (Ensemble Learning): Notebook compares Base Model vs RFC-selected features vs LOFO-derived by accuracy and concludes 'LOFO proved to be the most effective feature selection method in this comparison.'
- Build ML model and Conclusion (Introdution to ML and Classification Algorithms): k chosen by plotting accuracy vs k(1-40); KNN/SVM/DT compared by accuracy + classification_report; GridSearchCV/RandomizedSearchCV with scoring='accuracy'; LOFO feature importance.

Supporting content in these sessions: 32 reading, 22 in_class_quiz, 16 mcq

---

#### Preprocessing justified by the algorithm's needs  `ml`
A preprocessing step is not just applied but justified by a property of the algorithm that requires it.

Rules:
- State why the algorithm needs the preprocessing step, tying it to a model property.
- Do scaling/encoding/smoothing with an explicit reason, not as boilerplate.

Observed in 22 session(s): Apriori Implementation (Unsupervised Learning), Assumptions of Linear Regression Implementation (Supervised Learning), Build ML model and Conclusion (Introdution to ML and Classification Algorithms), DBSCAN (Unsupervised Learning), DBSCAN Implementation (Unsupervised Learning), Dimentionality Reduction Implementation (Unsupervised Learning), Dimentionality Reduction Part 2 (Unsupervised Learning), EDA (Introdution to ML and Classification Algorithms), Eclat Implementation (Unsupervised Learning), Gradient Descent (Supervised Learning), K-Means implementation (Unsupervised Learning), KNN Advanced (Introdution to ML and Classification Algorithms), Logistic Regression Implementation (Supervised Learning), Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms), Model Building -1 (Unsupervised Learning), Model Building 1 (Ensemble Learning), Model Building 2 (Ensemble Learning), Naive Bayes Part 2 (Introdution to ML and Classification Algorithms), Problem Statement (Unsupervised Learning), Regularization Implementation (Supervised Learning), SLR Implementation (Supervised Learning), Support Vector Regression Implementation (Supervised Learning)

Evidence:
- K-Means implementation (Unsupervised Learning): 'K-Means forms clusters using distance between points... If features are on different scales, the feature with larger numeric values will dominate the distance calculation, and the clustering result b
- Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms): 'We will use median to fill the null values as earlier we saw that loan amount have outliers so the mean will not be the proper approach as it is highly affected by the presence of outliers'; log tran
- Build ML model and Conclusion (Introdution to ML and Classification Algorithms): LoanAmount imputed with median not mean because 'loan amount have outliers so the mean will not be the proper approach'; income normalized because 'algorithm works better if the data is normally distr
- Eclat Implementation (Unsupervised Learning): Builds a vertical database item -> set of transaction IDs because ECLAT computes support via tidset intersections rather than rescanning.

Supporting content in these sessions: 22 reading, 12 in_class_quiz, 10 mcq

---

#### Interpret results back into the problem domain  `notebook`
An implementation closes by translating model output into domain terms and an actionable recommendation, not just a score.

Rules:
- Translate cluster IDs / predictions / metrics into domain-meaningful labels.
- End with an actionable statement tied to the original business problem.

Observed in 18 session(s): Apriori Algorithm (Unsupervised Learning), Apriori Implementation (Unsupervised Learning), Boosting Implementation - Classification (Ensemble Learning), Boosting Implementation - Regression (Ensemble Learning), Build ML model and Conclusion (Introdution to ML and Classification Algorithms), DBSCAN Implementation (Unsupervised Learning), Eclat (Unsupervised Learning), Eclat Implementation (Unsupervised Learning), Evaluation Metrics Implementation (Introdution to ML and Classification Algorithms), Feature Engineering - 2 (Introdution to ML and Classification Algorithms), Feature Importance Techniques (Ensemble Learning), K-Means implementation (Unsupervised Learning), Model Building -1 (Unsupervised Learning), Model Building 1 (Ensemble Learning), Model Building 2 (Ensemble Learning), Model Building 2 (Unsupervised Learning), Problem Statement (Unsupervised Learning), Random Forest Implementation (Ensemble Learning)

Evidence:
- K-Means implementation (Unsupervised Learning): Clusters labeled Budget/Careful/Premium/Impulsive; 'Premium customers who are the most valuable targets for high-end products'; 'can be used to design targeted marketing strategies.'
- Feature Importance Techniques (Ensemble Learning): Notebook summary maps metrics to churn business: recall 47% means 'many churn cases are being missed, which is critical in customer churn analysis.'
- Build ML model and Conclusion (Introdution to ML and Classification Algorithms): LOFO shows Credit_History is the most critical predictor; 'Actionable Insights Based on LOFO' gives retain/drop guidance.
- Eclat Implementation (Unsupervised Learning): Step 'Interpret buying patterns'; top rules by lift plotted as item-pair associations for retail buying patterns.

Supporting content in these sessions: 18 reading, 11 in_class_quiz, 5 mcq

---

#### Hyperparameter sweep to reveal effect and trade-offs  `notebook`
A small controlled/synthetic dataset is generated and the same model is fit multiple times, each time varying a single hyperparameter across contrasting values, so its isolated effect on the fitted model can be observed apart from the real modeling dataset.

Rules:
- Generate synthetic or controlled data for the demonstration rather than reusing the real project dataset
- Vary one hyperparameter at a time across clearly contrasting values (e.g. small vs large)
- Label the expected effect of each setting (e.g. overfitting vs better generalization, narrow vs wide margin)
- Present it as an illustrative aside, separate from the main load/train/evaluate pipeline

Observed in 8 session(s): DBSCAN Implementation (Unsupervised Learning), Dimentionality Reduction Implementation (Unsupervised Learning), Dimentionality Reduction Part 3 (Unsupervised Learning), Eclat Implementation (Unsupervised Learning), Model Building -1 (Unsupervised Learning), Model Building 2 (Unsupervised Learning), Problem Statement (Unsupervised Learning), Support Vector Regression Implementation (Supervised Learning)

Evidence:
- Eclat Implementation (Unsupervised Learning): 'Effect of min_support: run ECLAT for multiple support values' support_tests=[0.02,0.01,0.005,0.001], plotting itemset counts by k before choosing one.
- Problem Statement (Unsupervised Learning): K-Means 'for k in range(2, 11): ... print Silhouette, Dunn'; 'Visual Comparison of Different K Values' for candidate_ks=[2,3,4].
- Model Building -1 (Unsupervised Learning): 'K-Means Pipeline: Compute Metrics for K=2 to K=10' then plot the three metrics to select the optimal K.
- Support Vector Regression Implementation (Supervised Learning): A '## Visual Interpretation of SVR' section fits SVR on synthetic data (y = 2*x + 5 + noise) with contrasting settings: C=1000 vs C=0.1 (labeled 'overfitting' vs 'better generalization'), epsilon=0.1 

Supporting content in these sessions: 8 reading, 5 in_class_quiz, 4 mcq

---

#### Diagnose-then-remedy pairing  `ml`
A detected problem, violated assumption, or poor metric is never left as a bare diagnosis; it is immediately paired with one or more concrete corrective actions (transform data, add polynomial terms, regularize, drop a feature, change model) the learner can take.

Rules:
- Whenever a problem or violation is surfaced, list actionable remedies directly after it.
- Remedies are concrete named techniques, not vague advice.
- Applies to both slide guidance ('How to Check' + fix) and notebook interpretation cells ('Potential solutions' / 'Next Possible Steps').

Observed in 6 session(s): Assumptions of Linear Regression (Supervised Learning), Assumptions of Linear Regression Implementation (Supervised Learning), Build ML model and Conclusion (Introdution to ML and Classification Algorithms), Gradient Descent (Supervised Learning), Multiple Linear Regression (Supervised Learning), Polynomial Regression (Supervised Learning)

Evidence:
- Build ML model and Conclusion (Introdution to ML and Classification Algorithms): Right-skew from outliers diagnosed, then 'One way to remove the skewness is by doing the log transformation' applied to LoanAmount/ApplicantIncome.
- Assumptions of Linear Regression (Supervised Learning): Each assumption has a How-to-Check plus a fix: linearity 'Use transformations (e.g., log or polynomial terms) if non-linearity is detected'; VIF example 'remove X2 to address multicollinearity'.
- Multiple Linear Regression (Supervised Learning): After large MSE, '## Steps to Improve the Model:' lists Feature Engineering/Scaling/Outlier Treatment/regularization, then log-transforms charges and refits.
- Gradient Descent (Supervised Learning): High-cost diagnosis paired with fixes: 'Features may need additional scaling or transformation' and capturing non-linear relationships.

Supporting content in these sessions: 6 reading, 5 mcq, 3 in_class_quiz

---

#### Imputation/cleaning strategy justified by feature characteristics  `ml`
Missing-value and outlier handling is not applied with one blanket method; features are grouped by their statistical or semantic characteristics and each group is assigned a cleaning strategy explicitly justified by those characteristics.

Rules:
- Partition features into groups by type/behavior (e.g., continuous, categorical/clustered, time-series, strongly-correlated) before imputing.
- For each group, state why the chosen strategy fits it (e.g., mode for values that cluster around common levels, linear interpolation for gradually changing time-series, predictive or group-median for correlated features).
- Apply the matched per-group strategy rather than a single global imputation, and re-check missing counts after each step.

Observed in 5 session(s): Build Base ML model (Supervised Learning), Build Final ML model (Supervised Learning), Feature Engineering - 1 (Supervised Learning), Feature Engineering - 2 (Supervised Learning), Missing Values and Outliers Treatment (Supervised Learning)

Evidence:
- Feature Engineering - 2 (Supervised Learning): Grouped-imputation rationale block present in the notebook.
- Build Final ML model (Supervised Learning): Grouped-imputation rationale block present in the notebook.
- Feature Engineering - 1 (Supervised Learning): Same grouped-imputation rationale present (Mean/Median vs Mode vs Predictive vs Interpolation), each justified by feature type.
- Missing Values and Outliers Treatment (Supervised Learning): Features grouped Mean/Median, Mode, Predictive, Interpolation with per-group rationale: 'vaccination coverage percentages ... cluster around common values ... Mode reflects the most frequent'; 'time-s

Supporting content in these sessions: 5 reading

---

#### Algorithm stated as an explicit numbered step sequence  `ml`
An algorithm's mechanics are presented as an ordered, numbered list of discrete procedural steps (pseudocode-style) the learner follows in sequence. Distinct from 'Worked example computed step by step' because it describes the general procedure rather than plugging in concrete numbers.

Rules:
- Steps are explicitly numbered and in execution order.
- Describes the general procedure/mechanics, not a specific numeric computation.
- Often reused verbatim as a recap slide in later sessions that build on the algorithm.

Observed in 4 session(s): Hierarchical Clustering (Unsupervised Learning), K-Means Clustering (Unsupervised Learning), K-Means Clustering Part 3 (Unsupervised Learning), KNN (Introdution to ML and Classification Algorithms)

Evidence:
- Hierarchical Clustering (Unsupervised Learning): 'Steps: 01 Start with individual points / 02 Calculate distances between clusters / 03 Merge closest clusters / 04 Update distance matrix / 05 Repeat / 06 Create a dendrogram', each expanded on its ow
- K-Means Clustering (Unsupervised Learning): 'Working of K-Means: 01 Choose K / 02 Select initial centroids / 03 Assign data points / 04 Recalculate centroids / 05 Repeat step 3 and 4', each step then executed on its own slide.
- K-Means Clustering Part 3 (Unsupervised Learning): 'Steps to calculate K Means++: Select the first centroid randomly / Compute distance of every point / Assign higher importance to faraway points / Select next centroid / Repeat until K centroids chose
- KNN (Introdution to ML and Classification Algorithms): The 'Algorithm' slide lays out the KNN procedure as ordered steps 1-6: Plot training dataset, Locate test instance, Calculate distance from all train points, Sort ascending, Choose first k, Take mode 

Supporting content in these sessions: 4 reading, 4 in_class_quiz, 3 mcq

---

#### Hypothesis generation before analysis  `ml`
Before examining the data, the learner brainstorms expected relationships between features and the target from domain understanding, producing testable hypotheses that later EDA can confirm or refute.

Rules:
- State hypotheses before EDA, grounded in domain reasoning rather than the data itself
- Frame each hypothesis as an expected feature-to-target effect
- Revisit the hypotheses during or after EDA to confirm or refute them

Observed in 2 session(s): Problem Statement (Introdution to ML and Classification Algorithms), Problem Statement (Supervised Learning)

Evidence:
- Problem Statement (Supervised Learning): 'Hypothesis Generation... brainstorming potential factors that could influence the outcome of interest... before analyzing the data', with a domain rationale per factor: GDP -> 'Wealthier nations can 
- Problem Statement (Introdution to ML and Classification Algorithms): 'Hypothesis Generation' slide posits expected feature->target effects before EDA: 'Applicants with higher income are likely to have better chances of loan approval', 'Smaller loan amounts are expected

Supporting content in these sessions: 2 reading

---

#### Metric interpreted in plain-language real-world units  `ml`
Immediately after an error or goodness-of-fit metric is presented, a concrete value of it is translated into an intuitive plain-language sentence expressed in the target variable's real-world units or as a percentage, so the number carries meaning rather than staying abstract.

Rules:
- State a specific numeric value of the metric, not just its formula.
- Translate that value into a sentence in the domain's units (e.g. 'off by 4.91 rupees') or as a share of variance explained.
- Attach the interpretation right where the metric is defined.

Observed in 2 session(s): Gradient Descent Part - 2 (Supervised Learning), Simple Linear Regression (Supervised Learning)

Evidence:
- Gradient Descent Part - 2 (Supervised Learning): 'An R2 of 0.9536 means that 95.36% of the variation in sales is explained by the features'; Adjusted R2 0.9443 similarly interpreted.
- Simple Linear Regression (Supervised Learning): 'An MAE of 4.91 means that, on average, your predicted tip is off by 4.91 rs'; RMSE 'about 6.31 rs, in the same unit as tip'.

Supporting content in these sessions: 3 in_class_quiz, 3 mcq, 2 reading

---

#### Train/test hygiene to prevent data leakage  `ml`
Preprocessing and resampling are deliberately confined to training data and the test set is kept untouched, with the notebook/slide explicitly naming data leakage or metric inflation as the reason. Splits are stratified so class proportions are preserved and evaluation reflects the real-world distribution.

Rules:
- Fit transformers (scalers) and resamplers on the training split only, then apply to test with the fitted object.
- State the risk being avoided in plain language (data leakage, or metrics inflated by resampling).
- Keep the test set at its original, untouched distribution for evaluation.
- Use stratified splitting when classes are imbalanced so train and test keep the same class proportions.

Observed in 2 session(s): Model Building 1 (Ensemble Learning), Model Building 2 (Ensemble Learning)

Evidence:
- Model Building 1 (Ensemble Learning): 'Fit scaler ONLY on training data and transform both train and test'; 'stratify=y' with note 'train and test sets maintain the same class proportions.'
- Model Building 2 (Ensemble Learning): 'We apply oversampling only on training data. Test data remains untouched. This ensures no data leakage'; evaluation done 'strictly on the original, untouched test set... not inflated by resampling.'

Supporting content in these sessions: 2 reading

---

#### Baseline model before applying the technique  `ml`
Before applying the technique of interest, a simple reference model is trained on the raw/untransformed data to set a performance benchmark, and the technique's benefit is then measured as the change relative to that baseline.

Rules:
- Train a simple model on the original features first
- Record reference metrics such as accuracy and training time
- Report the technique's impact as a before/after comparison against the baseline

Observed in 1 session(s): Dimentionality Reduction Implementation (Unsupervised Learning)

Evidence:
- Dimentionality Reduction Implementation (Unsupervised Learning): Slide '04 - Build baseline model: Train a Logistic Regression classifier using original features to set a performance benchmark'; notebook prints 'Baseline (784 features) Accuracy'.

Supporting content in these sessions: 1 reading, 1 in_class_quiz

---

#### Investigate before dropping data  `notebook`
A data-quality issue (missing values, anomalous rows) is profiled and characterized with evidence before any rows are removed, and the keep/drop choice is stated as an explicit, justified Decision.

Rules:
- Profile the affected subset before altering the data
- Compare it against the retained data across several dimensions
- State an explicit Decision naming the reason for keeping or dropping

Observed in 1 session(s): EDA (Unsupervised Learning)

Evidence:
- EDA (Unsupervised Learning): 'Before dropping any rows, we investigate who these customers without IDs are ... could be guest buyers'; after a multi-plot comparison, 'Decision: We proceed with registered customers only.'

Supporting content in these sessions: 1 reading

---

#### Domain-driven feature engineering  `ml`
Raw records are aggregated or transformed into new features chosen for their domain meaning, and each engineered feature is explained in domain terms and by how it is computed.

Rules:
- Derive features motivated by the business/domain (e.g., RFM), not just raw columns
- State what each feature captures and how it is computed
- Aggregate to the correct analysis unit (e.g., one row per customer)

Observed in 1 session(s): EDA (Unsupervised Learning)

Evidence:
- EDA (Unsupervised Learning): Step 4 aggregates transactions to customer level building RFM features plus derived 'SpendRange' and 'ProductDiversity = UniqueProducts / TotalItems'; slide 'Feature Engineering: Aggregate transaction

Supporting content in these sessions: 1 reading

---

### Session-flow patterns

#### Agenda-first, Key-Takeaways-last framing  `universal`
Theory decks open with an Agenda slide listing the concepts in order and close with a Key Takeaways slide that mirrors the same items.

Rules:
- Open the deck with an Agenda slide naming the concepts in the sequence they will be taught.
- Close with a Key Takeaways slide that lists the same agenda items as a bare recap, introducing no new content.

Observed in 85 session(s): AUC - ROC Curve (Ensemble Learning), Apriori Algorithm (Unsupervised Learning), Apriori Implementation (Unsupervised Learning), Assumptions of Linear Regression (Supervised Learning), Bias and Variance (Introdution to ML and Classification Algorithms), Boosting Implementation - Classification (Ensemble Learning), Boosting Implementation - Regression (Ensemble Learning), Boosting methods and Adaboost (Ensemble Learning), Build Base ML model (Introdution to ML and Classification Algorithms), Build Base ML model (Supervised Learning), DA with Pandas Part - 1 (Introdution to ML and Classification Algorithms), DA with Pandas Part - 2 (Introdution to ML and Classification Algorithms), DBSCAN (Unsupervised Learning), DBSCAN Implementation (Unsupervised Learning), Decision Tree Implementation Part 1 (Introdution to ML and Classification Algorithms), Decision Tree Implementation Part 2 (Introdution to ML and Classification Algorithms), Decision Tree Part 1 (Introdution to ML and Classification Algorithms), Decision Tree Part 2 (Introdution to ML and Classification Algorithms), Decision Tree Regression (Supervised Learning), Decision Tree Regression Implementation (Supervised Learning), Dimentionality Reduction (Unsupervised Learning), Dimentionality Reduction Implementation (Unsupervised Learning), Dimentionality Reduction Part 2 (Unsupervised Learning), Dimentionality Reduction Part 3 (Unsupervised Learning), EDA (Ensemble Learning), EDA (Introdution to ML and Classification Algorithms), EDA (Supervised Learning), EDA (Unsupervised Learning), Eclat (Unsupervised Learning), Eclat Implementation (Unsupervised Learning), Evaluation Metrics (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Supervised Learning), Feature Engineering - 2 (Introdution to ML and Classification Algorithms), Feature Engineering - 2 (Supervised Learning), Feature Importance Techniques (Ensemble Learning), Gradient Boosting Classification (Ensemble Learning), Gradient Boosting Regression (Ensemble Learning), Gradient Descent Part - 1 (Supervised Learning), Gradient Descent Part - 2 (Supervised Learning), Hierarchical Clustering (Unsupervised Learning), Hierarchical Clustering Implementation (Unsupervised Learning), HyperParameter Tuning Part1 (Introdution to ML and Classification Algorithms), HyperParameter Tuning Part2 (Introdution to ML and Classification Algorithms), Introduction to Ensemble Algorithms (Ensemble Learning), Introduction to Machine Learning (Introdution to ML and Classification Algorithms), Introduction to Unsupervised Learning (Unsupervised Learning), K-Means Clustering (Unsupervised Learning), K-Means Clustering Part 2 (Unsupervised Learning), K-Means Clustering Part 3 (Unsupervised Learning), K-Means implementation (Unsupervised Learning), KNN (Introdution to ML and Classification Algorithms), KNN Advanced (Introdution to ML and Classification Algorithms), KNN Implementation with Scikit-Learn (Introdution to ML and Classification Algorithms), KNN Regression (Supervised Learning), Logistic Regression (Supervised Learning), Logistic Regression Implementation (Supervised Learning), Machine Learning Life Cycle (Introdution to ML and Classification Algorithms), Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms), Model Building -1 (Unsupervised Learning), Model Building 1 (Ensemble Learning), Model Building 2 (Ensemble Learning), Model Building 2 (Unsupervised Learning), Multiple Linear Regression (Supervised Learning), Naive Bayes Part 1 (Introdution to ML and Classification Algorithms), Naive Bayes Part 2 (Introdution to ML and Classification Algorithms), Polynomial Regression (Supervised Learning), Problem Statement (Ensemble Learning), Problem Statement (Introdution to ML and Classification Algorithms), Problem Statement (Supervised Learning), Problem Statement (Unsupervised Learning), Random Forest (Ensemble Learning), Random Forest Implementation (Ensemble Learning), Regularization (Supervised Learning), SLR Implementation (Supervised Learning), SVM (Introdution to ML and Classification Algorithms), SVM Implementation (Introdution to ML and Classification Algorithms), Setting Up ML Environment (Introdution to ML and Classification Algorithms), Simple Linear Regression (Supervised Learning), Stacking (Ensemble Learning), Support Vector Regression (Supervised Learning), Support Vector Regression Implementation (Supervised Learning), Voting (Ensemble Learning), XG boost (Ensemble Learning)

Evidence:
- SVM Implementation (Introdution to ML and Classification Algorithms): Opens with an 'Agenda' slide (SVM Implementation, Pros & Cons) and closes with 'Key Takeaways' listing Train test split, Label Encoder, Kernel Trick, Pros and Cons.
- K-Means implementation (Unsupervised Learning): Agenda: 'Problem statement, Pros and cons'; Key Takeaways: 'K-Means implementation, Pros and cons'.
- Evaluation Metrics (Introdution to ML and Classification Algorithms): Agenda lists Confusion Matrix, Accuracy, Precision, Recall, F1 Score; Key Takeaways mirrors them plus Accuracy Paradox.
- Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms): Agenda 'Missing Value Imputation / Outlier Treatment' mirrored by Key Takeaways 'Missing Values / Missing Value Imputation Methods / Outlier & Treatment Methods'.

Supporting content in these sessions: 84 reading, 61 in_class_quiz, 49 mcq

---

#### Recap bridge from prior session  `universal`
A continuation session begins by explicitly recapping what the previous session established, then states what this session adds.

Rules:
- Start a follow-on session with a short recap of the prior session's conclusion before adding new material.
- State explicitly what the current session will add on top of the recap.

Observed in 33 session(s): AUC - ROC Curve (Ensemble Learning), Bagging (Ensemble Learning), Bias and Variance (Introdution to ML and Classification Algorithms), DBSCAN (Unsupervised Learning), Decision Tree Implementation Part 2 (Introdution to ML and Classification Algorithms), Decision Tree Part 2 (Introdution to ML and Classification Algorithms), Decision Tree Regression (Supervised Learning), Eclat (Unsupervised Learning), Evaluation Metrics (Introdution to ML and Classification Algorithms), Feature Importance Techniques (Ensemble Learning), Gradient Boosting Classification (Ensemble Learning), Gradient Boosting Regression (Ensemble Learning), Hierarchical Clustering (Unsupervised Learning), HyperParameter Tuning Part1 (Introdution to ML and Classification Algorithms), HyperParameter Tuning Part2 (Introdution to ML and Classification Algorithms), Introduction to Ensemble Algorithms (Ensemble Learning), Introduction to Unsupervised Learning (Unsupervised Learning), K-Means Clustering Part 2 (Unsupervised Learning), K-Means Clustering Part 3 (Unsupervised Learning), KNN Advanced (Introdution to ML and Classification Algorithms), KNN Regression (Supervised Learning), Machine Learning Life Cycle (Introdution to ML and Classification Algorithms), Multiple Linear Regression (Supervised Learning), Naive Bayes Part 2 (Introdution to ML and Classification Algorithms), Polynomial Regression (Supervised Learning), Problem Statement (Introdution to ML and Classification Algorithms), Random Forest (Ensemble Learning), Setting Up ML Environment (Introdution to ML and Classification Algorithms), Simple Linear Regression (Supervised Learning), Stacking (Ensemble Learning), Support Vector Regression (Supervised Learning), Voting (Ensemble Learning), XG boost (Ensemble Learning)

Evidence:
- AUC - ROC Curve (Ensemble Learning): 'Recap - Confusion Matrix... Before moving to the AUC-ROC curve, let's quickly recap the confusion matrix.'
- Hierarchical Clustering (Unsupervised Learning): 'Recap of K-Means clustering... K-means tries to split the data into k groups based on similarity. But what if we don't know how many groups there are?'
- K-Means Clustering Part 2 (Unsupervised Learning): 'In previous session, we manually choose the number of clusters. But in real-world problems, we don't always know the correct value of K... we chose K using trial and visual inspection.'
- Feature Importance Techniques (Ensemble Learning): 'Recap: Random Forest builds many trees using random features and splits nodes using entropy and information gain'; repeated tag 'All these trees are built by using random forest (from previous sessio

Supporting content in these sessions: 33 reading, 31 in_class_quiz, 23 mcq

---

#### Multi-session capstone project pipeline  `ml`
A multi-session project opens with a roadmap slide listing every phase of the end-to-end workflow (problem statement, EDA, cleaning, feature engineering, modeling, conclusion), situating each individual session within the larger project arc.

Rules:
- Enumerate the full sequence of project phases, not just the current session's topics
- Present it at the project kickoff so learners see the whole arc
- Keep it distinct from the per-session Agenda slide, which scopes only that session

Observed in 5 session(s): EDA (Unsupervised Learning), Model Building -1 (Unsupervised Learning), Model Building 2 (Unsupervised Learning), Problem Statement (Introdution to ML and Classification Algorithms), Problem Statement (Unsupervised Learning)

Evidence:
- Problem Statement (Unsupervised Learning): Titled 'Capstone Project - Customer Segmentation'; closing speaker note 'Next session: EDA, data cleaning, and feature engineering.'
- Problem Statement (Introdution to ML and Classification Algorithms): 'Project Workflow' slide enumerates the whole capstone arc: Problem Statement / EDA / Missing Values and Outlier treatment / Feature Engineering-1 / Evaluation Metrics / Build Base ML Model / Feature 
- Model Building -1 (Unsupervised Learning): Titled 'Capstone Project - ModelBuilding1', continuing the shared customer-segmentation project.
- EDA (Unsupervised Learning): Titled 'Capstone Project - EDA', continuing the shared customer-segmentation project and dataset from the prior session.

Supporting content in these sessions: 5 reading

---

### Learning strategies

#### Worked example computed step by step  `ml`
A formula is taught by plugging concrete numbers through explicit numbered steps, showing each intermediate value and ending in the resulting decision.

Rules:
- Instantiate the formula with concrete numbers rather than leaving it symbolic.
- Break the computation into labelled steps and show every intermediate value.
- End with the concrete decision or classification the numbers produce.

Observed in 28 session(s): AUC - ROC Curve (Ensemble Learning), Apriori Algorithm (Unsupervised Learning), Assumptions of Linear Regression (Supervised Learning), Bagging (Ensemble Learning), Boosting methods and Adaboost (Ensemble Learning), DBSCAN (Unsupervised Learning), Decision Tree Part 1 (Introdution to ML and Classification Algorithms), Decision Tree Part 2 (Introdution to ML and Classification Algorithms), Decision Tree Regression (Supervised Learning), Dimentionality Reduction Part 2 (Unsupervised Learning), Eclat (Unsupervised Learning), Evaluation Metrics (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Supervised Learning), Gradient Boosting Classification (Ensemble Learning), Gradient Boosting Regression (Ensemble Learning), K-Means Clustering (Unsupervised Learning), K-Means Clustering Part 2 (Unsupervised Learning), K-Means Clustering Part 3 (Unsupervised Learning), KNN Advanced (Introdution to ML and Classification Algorithms), Logistic Regression (Supervised Learning), Multiple Linear Regression (Supervised Learning), Naive Bayes Part 1 (Introdution to ML and Classification Algorithms), Polynomial Regression (Supervised Learning), Problem Statement (Unsupervised Learning), Random Forest (Ensemble Learning), Simple Linear Regression (Supervised Learning), Voting (Ensemble Learning), XG boost (Ensemble Learning)

Evidence:
- Evaluation Metrics (Introdution to ML and Classification Algorithms): Accuracy = (100+9700)/10000 = 0.98; full example computes Accuracy 98%, Precision 67%, Recall 40%, F1 50% with explicit substitutions.
- AUC - ROC Curve (Ensemble Learning): 4-email example: Step1 probabilities, Step2 pick thresholds, Step3 threshold 0.60 gives TPR=2/2 and FPR=1/2, Step4 vary threshold, Final AUC=0.875.
- K-Means Clustering Part 2 (Unsupervised Learning): Silhouette for P1: a(P1)=(0.58+1.56+1.61)/3=1.25, b(P1)=(5.47+6.10+6.88+6.86+7.22)/5=6.51, s(P1)=(6.51-1.25)/6.51=0.81; average across 9 points = 0.82.
- Assumptions of Linear Regression (Supervised Learning): VIF example on house price: VIF_X1=2, VIF_X3=6, VIF_X2=12 -> Number of Bedrooms flagged high, decision to remove X2.

Supporting content in these sessions: 27 reading, 26 in_class_quiz, 22 mcq

---

#### Active recall via posed questions  `universal`
The learner is asked a question and prompted to reason before the answer is revealed, sometimes with an explicit pause-and-attempt instruction.

Rules:
- Pose the question to the learner before revealing the answer.
- Use rhetorical prompts ('why's that?', 'what do you think will happen?') to drive reasoning.
- Occasionally instruct the learner to pause and attempt or imagine the result.

Observed in 18 session(s): AUC - ROC Curve (Ensemble Learning), Bagging (Ensemble Learning), Boosting methods and Adaboost (Ensemble Learning), Feature Importance Techniques (Ensemble Learning), Gradient Descent Part - 1 (Supervised Learning), Hierarchical Clustering (Unsupervised Learning), HyperParameter Tuning Part1 (Introdution to ML and Classification Algorithms), Introduction to Ensemble Algorithms (Ensemble Learning), Introduction to Unsupervised Learning (Unsupervised Learning), K-Means Clustering Part 2 (Unsupervised Learning), K-Means Clustering Part 3 (Unsupervised Learning), KNN Implementation with Scikit-Learn (Introdution to ML and Classification Algorithms), Logistic Regression (Supervised Learning), Naive Bayes Part 1 (Introdution to ML and Classification Algorithms), Problem Statement (Ensemble Learning), Problem Statement (Unsupervised Learning), Random Forest (Ensemble Learning), SVM (Introdution to ML and Classification Algorithms)

Evidence:
- AUC - ROC Curve (Ensemble Learning): 'the next question is: how do we measure its overall performance using a single number? That single number is AUC.'
- Hierarchical Clustering (Unsupervised Learning): 'But what if we don't know how many groups there are?' posed before introducing hierarchical clustering.
- K-Means Clustering Part 2 (Unsupervised Learning): Slide poses 'Which value of K is better?' before revealing the elbow selection.
- Feature Importance Techniques (Ensemble Learning): 'But across the entire algorithm, which features actually reduced uncertainty the most?'; 'LOFO answers: How much does the model rely on this feature to make accurate predictions?'

Supporting content in these sessions: 17 reading, 16 in_class_quiz, 13 mcq

---

### Core principles

#### Build the concept by fixing the previous version's limitation  `ml`
Teach the simplest form first, expose a concrete failure of it, then introduce the enhancement that repairs that failure, chaining ideas so each is motivated by a prior shortcoming.

Rules:
- Present the naive/simplest version of the method first.
- Show a concrete case where it breaks, then introduce the refinement that fixes exactly that break.
- Chain refinements so each new idea answers a limitation just demonstrated.

Observed in 33 session(s): AUC - ROC Curve (Ensemble Learning), Bagging (Ensemble Learning), Bias and Variance (Introdution to ML and Classification Algorithms), Boosting methods and Adaboost (Ensemble Learning), DBSCAN (Unsupervised Learning), DBSCAN Implementation (Unsupervised Learning), Dimentionality Reduction Part 3 (Unsupervised Learning), Eclat (Unsupervised Learning), Evaluation Metrics (Introdution to ML and Classification Algorithms), Feature Engineering - 2 (Supervised Learning), Feature Importance Techniques (Ensemble Learning), Gradient Boosting Regression (Ensemble Learning), Gradient Descent Part - 2 (Supervised Learning), Hierarchical Clustering (Unsupervised Learning), Hierarchical Clustering Implementation (Unsupervised Learning), HyperParameter Tuning Part1 (Introdution to ML and Classification Algorithms), Introduction to Ensemble Algorithms (Ensemble Learning), K-Means Clustering Part 2 (Unsupervised Learning), K-Means Clustering Part 3 (Unsupervised Learning), KNN Advanced (Introdution to ML and Classification Algorithms), Logistic Regression (Supervised Learning), Model Building 1 (Ensemble Learning), Model Building 2 (Ensemble Learning), Model Building 2 (Unsupervised Learning), Multiple Linear Regression (Supervised Learning), Naive Bayes Part 1 (Introdution to ML and Classification Algorithms), Polynomial Regression (Supervised Learning), Random Forest (Ensemble Learning), Regularization (Supervised Learning), SVM (Introdution to ML and Classification Algorithms), Simple Linear Regression (Supervised Learning), Stacking (Ensemble Learning), XG boost (Ensemble Learning)

Evidence:
- Evaluation Metrics (Introdution to ML and Classification Algorithms): Accuracy Paradox exposes that 98% accuracy can miss the spam class entirely, motivating Precision and Recall, then F1 as their harmonic-mean balance.
- AUC - ROC Curve (Ensemble Learning): ROC's imbalance weakness motivates PR curve: repeated 'We can overcome this by using PR curve', which plots Precision vs Recall to focus on the positive class.
- Hierarchical Clustering (Unsupervised Learning): 'K-Means assumes round, equal-sized clusters - real data often isn't like that. Hierarchical clustering doesn't make these assumptions'; 'We don't need to pick a number of clusters in advance.'
- K-Means Clustering Part 2 (Unsupervised Learning): Manual K 'not reliable for real-world' -> Elbow Method; then 'Sometimes the elbow point is not very clear... To handle such situations, we use another evaluation technique called the Silhouette Method

Supporting content in these sessions: 32 reading, 29 in_class_quiz, 25 mcq

---

#### Analogy or concrete scenario before formalism  `ml`
A new algorithm is introduced through an everyday analogy or a concrete scenario before any math, notation, or formal definition is shown.

Rules:
- Lead with a relatable analogy or a small concrete scenario that captures the algorithm's core idea.
- Only after the intuition lands, present the formal definition, terminology, or equations.

Observed in 22 session(s): Apriori Algorithm (Unsupervised Learning), Assumptions of Linear Regression (Supervised Learning), Bias and Variance (Introdution to ML and Classification Algorithms), Boosting methods and Adaboost (Ensemble Learning), Decision Tree Part 1 (Introdution to ML and Classification Algorithms), Dimentionality Reduction (Unsupervised Learning), Dimentionality Reduction Part 2 (Unsupervised Learning), Evaluation Metrics (Introdution to ML and Classification Algorithms), Feature Engineering - 2 (Introdution to ML and Classification Algorithms), Feature Importance Techniques (Ensemble Learning), Gradient Descent Part - 1 (Supervised Learning), Hierarchical Clustering (Unsupervised Learning), Introduction to Ensemble Algorithms (Ensemble Learning), KNN (Introdution to ML and Classification Algorithms), KNN Regression (Supervised Learning), Logistic Regression (Supervised Learning), Naive Bayes Part 1 (Introdution to ML and Classification Algorithms), Problem Statement (Unsupervised Learning), SVM (Introdution to ML and Classification Algorithms), Simple Linear Regression (Supervised Learning), Support Vector Regression (Supervised Learning), Voting (Ensemble Learning)

Evidence:
- Evaluation Metrics (Introdution to ML and Classification Algorithms): 'Imagine you work for a tech company that develops an email filtering system...' scenario opens the deck before the confusion matrix is defined.
- Hierarchical Clustering (Unsupervised Learning): 'This is like grouping friends in a room. First, people closest to each other form pairs'; 'A dendrogram is like a family tree for clusters.'
- Dimentionality Reduction (Unsupervised Learning): 'Real-World Analogy: choosing a phone using Battery/Camera/Price (easy) vs 200 specifications (confusing)' before the formal 'Curse of Dimensionality' definition.
- Feature Importance Techniques (Ensemble Learning): Cricket-team analogy for LOFO: remove opening batsman/fast bowler/fielder and see the drop; 'In LOFO, features are players, and model performance is the match result' before the formal Working-of-LOFO

Supporting content in these sessions: 21 reading, 20 in_class_quiz, 17 mcq

---

#### Flag optional or simplified mathematical depth  `ml`
Deeper math is explicitly marked as optional or as a deliberate simplification chosen for teaching convenience, so learners can engage with intuition without being blocked by rigor.

Rules:
- Mark advanced math as optional so intuition-only learners are not blocked.
- When simplifying for teaching, say so and explain the simplification's purpose.

Observed in 11 session(s): Boosting methods and Adaboost (Ensemble Learning), DBSCAN (Unsupervised Learning), Dimentionality Reduction Implementation (Unsupervised Learning), Eclat Implementation (Unsupervised Learning), Gradient Boosting Regression (Ensemble Learning), Gradient Descent Part - 1 (Supervised Learning), K-Means Clustering (Unsupervised Learning), K-Means implementation (Unsupervised Learning), Model Building 2 (Unsupervised Learning), SVM (Introdution to ML and Classification Algorithms), XG boost (Ensemble Learning)

Evidence:
- K-Means implementation (Unsupervised Learning): Selects only 2 features to 'allow a clean 2D cluster plot'; 'Note: If we include more features... we can use PCA to reduce them into 2 principal components.'
- Eclat Implementation (Unsupervised Learning): In generate_rules: '# try all single-item consequents for simplicity (easy to learn)' - deliberate simplification of rule generation for teaching.
- XG boost (Ensemble Learning): Explicit simplifications: 'For simplicity, let's assume the learning rate is 1 (no scaling)'; 'we'll assume that regularization doesn't change the updated predictions in this case'; an authoring note 
- Gradient Descent Part - 1 (Supervised Learning): 'Gradient Descent is a general optimization algorithm... but for mathematical convenience, we are learning it with Linear Regression'; MSE scaled by 1/2 'to avoid carrying this extra 2... simplifying 

Supporting content in these sessions: 11 reading, 10 in_class_quiz, 7 mcq

---

#### Valid-case-versus-violation contrast  `ml`
A concept, assumption, or required condition is taught by pairing a concrete example that satisfies it (valid case) with a concrete example that breaks it (violation), defining the boundary of the concept through both a positive and a negative example.

Rules:
- For each concept give one satisfying example and one violating example.
- Keep the two examples concrete and parallel so the difference is the concept itself.
- State the desired/valid outcome explicitly so the violation is recognizable.

Observed in 2 session(s): Assumptions of Linear Regression (Supervised Learning), Assumptions of Linear Regression Implementation (Supervised Learning)

Evidence:
- Assumptions of Linear Regression (Supervised Learning): Every assumption pairs a Valid Case with a Violation, e.g. multicollinearity valid 'Height and age... correlation is low' vs violation 'Height in inches and height in centimeters... perfectly correlat
- Assumptions of Linear Regression Implementation (Supervised Learning): States the desired outcome to contrast against the observed violation: 'points are symmetrically distributed around a diagonal line' and 'The orange line should be flat' vs the actual curved line.

Supporting content in these sessions: 2 reading, 2 mcq, 1 in_class_quiz

---

#### Forward reference to upcoming content  `universal`
The material names a concept, technique, or algorithm relevant at the current point but explicitly defers its full treatment to a later session, so learners know it exists without being blocked by depth they do not need yet.

Rules:
- Only tag when the text explicitly signals deferral (e.g. 'we will learn later', 'covered later', 'more on this in a future session').
- Distinct from flagging optional/simplified math depth: here a named topic is postponed to another session rather than simplified in place.
- The forward-referenced item must be named concretely (a specific method/technique), not vaguely gestured at.

Observed in 1 session(s): Bias and Variance (Introdution to ML and Classification Algorithms)

Evidence:
- Bias and Variance (Introdution to ML and Classification Algorithms): Named-but-deferred topics: 'Random Forests or Gradient Boosting... (we will learn later)', 'PCA, we will learn later', 'L1 (Lasso) and L2 (Ridge) regularization... (We will learn later)'.

Supporting content in these sessions: 1 reading, 1 in_class_quiz, 1 mcq

---

#### Myth-versus-reality misconception correction  `universal`
A widely held but incorrect learner intuition is stated explicitly (often labelled as the common belief), then immediately contradicted with the correct behaviour, using the gap to motivate the topic.

Rules:
- State the misconception the way a learner would phrase it before correcting it.
- Explicitly contrast the belief with what actually happens.
- Use the corrected understanding to motivate the concept that follows.

Observed in 1 session(s): Dimentionality Reduction (Unsupervised Learning)

Evidence:
- Dimentionality Reduction (Unsupervised Learning): 'Common intuition: More features -> More information -> Better accuracy' immediately corrected 'Reality: More features can actually confuse the model'.

Supporting content in these sessions: 1 reading, 1 in_class_quiz, 1 mcq

---

### Cross-cutting patterns

#### Definition callouts for terminology  `ml`
Each new term gets a short self-contained definition, often clustered into a dedicated terminology segment.

Rules:
- Give every newly introduced term a concise standalone definition.
- Group foundational terms into one 'Terminologies' segment where several are introduced together.

Observed in 59 session(s): AUC - ROC Curve (Ensemble Learning), Apriori Algorithm (Unsupervised Learning), Apriori Implementation (Unsupervised Learning), Assumptions of Linear Regression (Supervised Learning), Bagging (Ensemble Learning), Bias and Variance (Introdution to ML and Classification Algorithms), Boosting methods and Adaboost (Ensemble Learning), DBSCAN (Unsupervised Learning), DBSCAN Implementation (Unsupervised Learning), Decision Tree Implementation Part 2 (Introdution to ML and Classification Algorithms), Decision Tree Part 1 (Introdution to ML and Classification Algorithms), Decision Tree Regression (Supervised Learning), Dimentionality Reduction (Unsupervised Learning), Dimentionality Reduction Part 2 (Unsupervised Learning), Dimentionality Reduction Part 3 (Unsupervised Learning), EDA (Ensemble Learning), EDA (Introdution to ML and Classification Algorithms), EDA (Supervised Learning), EDA (Unsupervised Learning), Eclat (Unsupervised Learning), Evaluation Metrics (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Supervised Learning), Feature Engineering - 2 (Introdution to ML and Classification Algorithms), Feature Engineering - 2 (Supervised Learning), Feature Importance Techniques (Ensemble Learning), Gradient Boosting Classification (Ensemble Learning), Gradient Boosting Regression (Ensemble Learning), Gradient Descent Part - 1 (Supervised Learning), Gradient Descent Part - 2 (Supervised Learning), Hierarchical Clustering (Unsupervised Learning), Hierarchical Clustering Implementation (Unsupervised Learning), HyperParameter Tuning Part1 (Introdution to ML and Classification Algorithms), Introduction to Ensemble Algorithms (Ensemble Learning), Introduction to Machine Learning (Introdution to ML and Classification Algorithms), Introduction to Unsupervised Learning (Unsupervised Learning), K-Means Clustering (Unsupervised Learning), K-Means Clustering Part 2 (Unsupervised Learning), K-Means Clustering Part 3 (Unsupervised Learning), K-Means implementation (Unsupervised Learning), KNN (Introdution to ML and Classification Algorithms), KNN Advanced (Introdution to ML and Classification Algorithms), Logistic Regression (Supervised Learning), Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms), Model Building 2 (Ensemble Learning), Model Building 2 (Unsupervised Learning), Multiple Linear Regression (Supervised Learning), Naive Bayes Part 1 (Introdution to ML and Classification Algorithms), Polynomial Regression (Supervised Learning), Problem Statement (Ensemble Learning), Problem Statement (Introdution to ML and Classification Algorithms), Problem Statement (Supervised Learning), Problem Statement (Unsupervised Learning), Random Forest (Ensemble Learning), Regularization (Supervised Learning), Simple Linear Regression (Supervised Learning), Stacking (Ensemble Learning), Voting (Ensemble Learning), XG boost (Ensemble Learning)

Evidence:
- K-Means implementation (Unsupervised Learning): '### Silhouette Score: It measures how well each data point fits within its assigned cluster'; '### Dunn Index: ratio of the minimum inter-cluster distance to the maximum intra-cluster diameter.'
- Evaluation Metrics (Introdution to ML and Classification Algorithms): TP/FN/FP/TN each defined in a 'Confusion Matrix Terminology' cluster; Accuracy, Precision, Recall, F1 each given a one-line definition.
- Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms): 'Missing value imputation is the process of replacing missing or null values...' and 'Outlier are data points that deviate significantly from the other observations in the dataset.'
- AUC - ROC Curve (Ensemble Learning): 'Terminology' segment; TP/FN/FP/TN each defined; 'What is AUC-ROC'; 'Random Classifier: A theoretical classifier that assigns scores completely at random.'

Supporting content in these sessions: 58 reading, 44 in_class_quiz, 37 mcq

---

#### Small hand-computable toy dataset to teach mechanics  `ml`
An algorithm's internal mechanics are illustrated on a tiny invented dataset small enough to trace by hand, kept separate from the larger real dataset used in the notebook.

Rules:
- Use a handful of rows/objects that can be counted and computed manually to expose the mechanics.
- Keep this toy example distinct from the realistic dataset used for implementation.

Observed in 31 session(s): AUC - ROC Curve (Ensemble Learning), Apriori Algorithm (Unsupervised Learning), Bagging (Ensemble Learning), Boosting methods and Adaboost (Ensemble Learning), DBSCAN (Unsupervised Learning), Decision Tree Part 1 (Introdution to ML and Classification Algorithms), Decision Tree Part 2 (Introdution to ML and Classification Algorithms), Decision Tree Regression (Supervised Learning), Dimentionality Reduction Part 2 (Unsupervised Learning), Eclat (Unsupervised Learning), Evaluation Metrics (Introdution to ML and Classification Algorithms), Evaluation Metrics Implementation (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Supervised Learning), Feature Importance Techniques (Ensemble Learning), Gradient Boosting Classification (Ensemble Learning), Gradient Boosting Regression (Ensemble Learning), Hierarchical Clustering (Unsupervised Learning), K-Means Clustering (Unsupervised Learning), K-Means Clustering Part 2 (Unsupervised Learning), K-Means Clustering Part 3 (Unsupervised Learning), KNN Advanced (Introdution to ML and Classification Algorithms), Multiple Linear Regression (Supervised Learning), Naive Bayes Implementation (Introdution to ML and Classification Algorithms), Naive Bayes Part 1 (Introdution to ML and Classification Algorithms), Naive Bayes Part 2 (Introdution to ML and Classification Algorithms), Polynomial Regression (Supervised Learning), Random Forest (Ensemble Learning), Simple Linear Regression (Supervised Learning), Voting (Ensemble Learning), XG boost (Ensemble Learning)

Evidence:
- Evaluation Metrics (Introdution to ML and Classification Algorithms): Invented confusion-matrix counts (100 TP, 150 FN, 50 FP, 9700 TN) used to compute every metric by hand.
- AUC - ROC Curve (Ensemble Learning): 'Suppose we have a spam detection model and only 4 emails' (A,B,C,D) with predicted probabilities 0.90/0.60/0.65/0.20.
- Hierarchical Clustering (Unsupervised Learning): Four fruits (Apple 100g, Banana 120g, Cherry 50g, Grape 30g) merged step by step: 'Grape and Cherry are closest -> Merge', then Apple+Banana, then all.
- K-Means Clustering Part 2 (Unsupervised Learning): 9-point invented set P1-P9 with Feature 1 / Feature 2 used to hand-compute silhouette scores.

Supporting content in these sessions: 30 reading, 29 in_class_quiz, 23 mcq

---

#### Term-by-term decomposition of a formula  `ml`
After a formula is shown, each symbol is glossed one at a time in plain language.

Rules:
- Present the whole formula, then annotate each symbol individually with a plain-language meaning.
- Give each term its own explanation rather than glossing the equation as a block.

Observed in 25 session(s): AUC - ROC Curve (Ensemble Learning), Apriori Algorithm (Unsupervised Learning), Assumptions of Linear Regression (Supervised Learning), Boosting methods and Adaboost (Ensemble Learning), Decision Tree Part 2 (Introdution to ML and Classification Algorithms), Dimentionality Reduction Part 2 (Unsupervised Learning), Feature Engineering - 1 (Supervised Learning), Feature Engineering - 2 (Supervised Learning), Gradient Boosting Classification (Ensemble Learning), Gradient Boosting Regression (Ensemble Learning), Gradient Descent Part - 1 (Supervised Learning), Gradient Descent Part - 2 (Supervised Learning), Hierarchical Clustering (Unsupervised Learning), K-Means Clustering (Unsupervised Learning), K-Means Clustering Part 2 (Unsupervised Learning), K-Means Clustering Part 3 (Unsupervised Learning), KNN Advanced (Introdution to ML and Classification Algorithms), Logistic Regression (Supervised Learning), Multiple Linear Regression (Supervised Learning), Naive Bayes Part 1 (Introdution to ML and Classification Algorithms), Polynomial Regression (Supervised Learning), Problem Statement (Unsupervised Learning), Random Forest (Ensemble Learning), Regularization (Supervised Learning), Simple Linear Regression (Supervised Learning)

Evidence:
- AUC - ROC Curve (Ensemble Learning): TPR = TP/(TP+FN) with TP='Correctly predicted positives' and FN='Actual positives predicted as negative' glossed separately; FPR likewise; trapezoidal AUC formula broken down.
- Hierarchical Clustering (Unsupervised Learning): Ward's 'Delta = (nA*nB)/(nA+nB) * ||muA - muB||^2' glossed: 'nA, nB = number of points in each cluster; muA, muB = centroids; ||muA - muB||^2 = squared distance between centroids.'
- K-Means Clustering Part 2 (Unsupervised Learning): 'Silhouette Score = (b - a)/max(a,b)' with a = 'Average distance to other points in the same cluster (Intra-Cluster)', b = 'Average distance to points in the nearest neighboring cluster (Inter-Cluster
- Assumptions of Linear Regression (Supervised Learning): VIF = 1/(1 - Ri^2) explained: 'Ri^2 is the coefficient of determination obtained when Xi is regressed on all other independent variables.'

Supporting content in these sessions: 24 reading, 23 in_class_quiz, 20 mcq

---

#### Paired theory-plus-implementation with live-demo cues  `ml`
Slide decks carry authoring cues to demonstrate a point live or hand off to a notebook, revealing a design where concepts are narrated on slides and then applied hands-on.

Rules:
- Design the concept on slides and the hands-on application in a paired notebook.
- Mark points to demonstrate visually/live or to switch to the notebook rather than only telling.

Observed in 19 session(s): Bagging (Ensemble Learning), Bias and Variance (Introdution to ML and Classification Algorithms), DBSCAN (Unsupervised Learning), Decision Tree Regression (Supervised Learning), Dimentionality Reduction Part 2 (Unsupervised Learning), EDA (Ensemble Learning), EDA (Introdution to ML and Classification Algorithms), Gradient Boosting Regression (Ensemble Learning), HyperParameter Tuning Part1 (Introdution to ML and Classification Algorithms), Introduction to Ensemble Algorithms (Ensemble Learning), KNN Advanced (Introdution to ML and Classification Algorithms), KNN Regression (Supervised Learning), Logistic Regression (Supervised Learning), Model Building 2 (Ensemble Learning), SLR Implementation (Supervised Learning), SVM (Introdution to ML and Classification Algorithms), Simple Linear Regression (Supervised Learning), Stacking (Ensemble Learning), Voting (Ensemble Learning)

Evidence:
- Stacking (Ensemble Learning): Authoring cue on the Recap slide: 'need to explain this on the top of voting how stacking is different.'
- EDA (Introdution to ML and Classification Algorithms): Final slide carries an explicit handoff cue: 'Move to colab and explain class imbalance', with the EDA notebook doing the hands-on work.
- Logistic Regression (Supervised Learning): Authoring cues embedded: 'Explain this with tablet.', 'Explain the Equation on writing pad.', 'Show with the data', 'Mention threshold'.
- Voting (Ensemble Learning): Slide cue 'Move to colab after this' hands off from theory to the notebook; also 'Explain how it predicts the final output for classification and regression.'

Supporting content in these sessions: 19 reading, 16 in_class_quiz, 14 mcq

---

#### Preempt a look-alike confusion with explicit contrast  `universal`
A newly introduced concept is explicitly distinguished from a similar-sounding or previously-taught concept it is likely to be confused with, using a direct 'this is not that' / 'do not confuse' / 'same as X except...' warning aimed at heading off a specific misconception, rather than a neutral when-to-use comparison.

Rules:
- Trigger only on an explicit conflation warning (e.g. 'don't get confused', 'this is not X', 'same algorithm as X, only Y changes') whose purpose is preventing a specific misunderstanding.
- Distinct from 'Comparison table contrasting methods', whose purpose is when-to-use guidance across shared dimensions.
- In the evidence, name the two concepts being disambiguated and the confusion being headed off.

Observed in 2 session(s): Gradient Boosting Classification (Ensemble Learning), Gradient Boosting Regression (Ensemble Learning)

Evidence:
- Gradient Boosting Regression (Ensemble Learning): 'Using Gradient Boost for regression is different from doing linear regression, though both methods are related don't get confused' -- an explicit warning against conflating GB regression with linear 
- Gradient Boosting Classification (Ensemble Learning): 'Gradient Boosting Classification is the same algorithm as Gradient Boosting Regression, only the loss function and output interpretation change' -- heads off treating it as a different algorithm; als

Supporting content in these sessions: 2 reading, 2 in_class_quiz, 2 mcq

---

### Notebook / implementation patterns

#### Teaching code carries inline explanatory comments  `notebook`
Notebook code is annotated with short inline comments explaining what each nontrivial line does, so the code reads as a lesson.

Rules:
- Add a brief inline comment on nontrivial lines explaining intent and the method used.
- Keep comments teaching-oriented (what/why), not just restating syntax.

Observed in 48 session(s): Apriori Implementation (Unsupervised Learning), Assumptions of Linear Regression Implementation (Supervised Learning), Bagging (Ensemble Learning), Boosting Implementation - Classification (Ensemble Learning), Boosting Implementation - Regression (Ensemble Learning), Build Base ML model (Introdution to ML and Classification Algorithms), Build Base ML model (Supervised Learning), Build Final ML model (Supervised Learning), Build ML model and Conclusion (Introdution to ML and Classification Algorithms), DA with Pandas Part - 1 (Introdution to ML and Classification Algorithms), DA with Pandas Part - 2 (Introdution to ML and Classification Algorithms), DBSCAN Implementation (Unsupervised Learning), Decision Tree Regression Implementation (Supervised Learning), Dimentionality Reduction Implementation (Unsupervised Learning), EDA (Ensemble Learning), EDA (Introdution to ML and Classification Algorithms), EDA (Supervised Learning), EDA (Unsupervised Learning), Eclat Implementation (Unsupervised Learning), Evaluation Metrics Implementation (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Supervised Learning), Feature Engineering - 2 (Introdution to ML and Classification Algorithms), Feature Engineering - 2 (Supervised Learning), Feature Importance Techniques (Ensemble Learning), Gradient Descent (Supervised Learning), Hierarchical Clustering Implementation (Unsupervised Learning), HyperParameter Tuning Part1 (Introdution to ML and Classification Algorithms), HyperParameter Tuning Part2 (Introdution to ML and Classification Algorithms), K-Means implementation (Unsupervised Learning), KNN Implementation with Scikit-Learn (Introdution to ML and Classification Algorithms), KNN Regression (Supervised Learning), Logistic Regression Implementation (Supervised Learning), Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms), Missing Values and Outliers Treatment (Supervised Learning), Model Building 1 (Ensemble Learning), Model Building 2 (Ensemble Learning), Model Building 2 (Unsupervised Learning), Multiple Linear Regression (Supervised Learning), Naive Bayes Implementation (Introdution to ML and Classification Algorithms), Problem Statement (Unsupervised Learning), Random Forest Implementation (Ensemble Learning), Regularization Implementation (Supervised Learning), SLR Implementation (Supervised Learning), SVM (Introdution to ML and Classification Algorithms), Stacking (Ensemble Learning), Support Vector Regression Implementation (Supervised Learning), Voting (Ensemble Learning)

Evidence:
- K-Means implementation (Unsupervised Learning): '# Max intra-cluster distance', '# Min inter-cluster distance', '# Compute Silhouette Score (for chosen optimal k)'.
- Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms): Inline comments such as '# print data types for each variable' and prose annotations accompany each imputation cell.
- DA with Pandas Part - 1 (Introdution to ML and Classification Algorithms): Nearly every line commented, e.g. '# countplot gives us count of the values', '# Using .unique() we can find the unique values from column', '# lambda function to replace /5 to empty string'.
- DA with Pandas Part - 2 (Introdution to ML and Classification Algorithms): Inline comments throughout, e.g. '# checking column having null values or not using .isnull()', '# dropping duplicates values using .drop_duplicates()'.

Supporting content in these sessions: 48 reading, 26 in_class_quiz, 23 mcq

---

#### Systematic feature-by-feature / column-by-column pass  `notebook`
The notebook iterates over every feature or column in turn with uniform treatment, making the sweep exhaustive and consistent.

Rules:
- Iterate over every feature/column applying the same treatment (plot+interpret, or inspect+decide).
- State explicitly when a column needs no cleaning, so the pass is visibly exhaustive.

Observed in 23 session(s): Assumptions of Linear Regression Implementation (Supervised Learning), Bagging (Ensemble Learning), Build Base ML model (Introdution to ML and Classification Algorithms), Build Base ML model (Supervised Learning), Build Final ML model (Supervised Learning), Build ML model and Conclusion (Introdution to ML and Classification Algorithms), DA with Pandas Part - 1 (Introdution to ML and Classification Algorithms), DA with Pandas Part - 2 (Introdution to ML and Classification Algorithms), EDA (Introdution to ML and Classification Algorithms), EDA (Supervised Learning), EDA (Unsupervised Learning), Feature Engineering - 1 (Introdution to ML and Classification Algorithms), Feature Engineering - 1 (Supervised Learning), Feature Engineering - 2 (Introdution to ML and Classification Algorithms), Feature Engineering - 2 (Supervised Learning), Gradient Descent (Supervised Learning), Logistic Regression Implementation (Supervised Learning), Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms), Missing Values and Outliers Treatment (Supervised Learning), Multiple Linear Regression (Supervised Learning), Problem Statement (Introdution to ML and Classification Algorithms), Regularization Implementation (Supervised Learning), SVM (Introdution to ML and Classification Algorithms)

Evidence:
- Missing Values and Outliers Treatment (Introdution to ML and Classification Algorithms): Missing values imputed feature by feature (Gender, Married, Dependents, Self_Employed, Credit_History by mode; Loan_Amount_Term by mode; LoanAmount by median) with the same treatment mirrored on the t
- DA with Pandas Part - 1 (Introdution to ML and Classification Algorithms): 'Cleaning Individual Columns' iterates uniformly over restaurants, online_order, book_table, rating, votes, location, rest_type, food_type, cost, type, city, each checked with .unique() and cleaned on
- DA with Pandas Part - 2 (Introdution to ML and Classification Algorithms): 'Cleaning Individual Columns' sweeps restaurants, online_order, book_table, rating, votes, location, rest_type, food_type, cost, type, city with uniform treatment.
- Build ML model and Conclusion (Introdution to ML and Classification Algorithms): Univariate analysis walks Gender/Married/Self_Employed/Credit_History then imputes missing values feature by feature.

Supporting content in these sessions: 23 reading, 9 mcq, 6 in_class_quiz

---

#### Question-driven exploratory analysis  `notebook`
An EDA notebook states an explicit upfront list of concrete analytical questions, then works through them one by one, answering each with a plot plus interpretation in the same order. Proactive question framing that drives the analysis, distinct from the reactive per-output interpretation of the observe-and-interpret loop.

Rules:
- A numbered or bulleted list of analytical questions is stated before the analysis begins.
- Each question is later answered in the same order with a computation/plot and a short interpretation.
- Questions are phrased in domain terms ('Find best location', 'most profitable type of restaurant').

Observed in 2 session(s): DA with Pandas Part - 1 (Introdution to ML and Classification Algorithms), DA with Pandas Part - 2 (Introdution to ML and Classification Algorithms)

Evidence:
- DA with Pandas Part - 1 (Introdution to ML and Classification Algorithms): '## My Findings' lists 10 concrete questions upfront (e.g. 'How many restaurants accepting online order?', 'Find best location'), then the Data Visualization section answers each in the same numbered 
- DA with Pandas Part - 2 (Introdution to ML and Classification Algorithms): '## My Findings' poses 10 analytical questions upfront, then the Data Visualization section answers them numbered 1)-10).

Supporting content in these sessions: 2 reading, 2 mcq, 1 in_class_quiz

---

#### Expose the fitted model's internals to demystify the abstraction  `notebook`
A high-level library model (especially an ensemble) is not left as a black box; the notebook reaches inside the fitted object or manually reconstructs the intermediate representation it consumes, then interprets what the exposed internals reveal about how the method works.

Rules:
- After fitting a composite estimator, inspect its constituent parts (e.g., individual base estimators via estimators_) or manually reproduce its internal intermediate data (e.g., stacking meta-features).
- Pair the inspection with a short written interpretation of what the internals show (e.g., why the pieces differ, or how they feed the next stage).
- Use the real fitted model on the actual dataset, not a separate toy stand-in.

Observed in 2 session(s): Bagging (Ensemble Learning), Stacking (Ensemble Learning)

Evidence:
- Stacking (Ensemble Learning): Before calling StackingClassifier, the notebook manually builds meta-features from knn_proba/dt_proba via np.hstack into a named DataFrame printed as 'Meta-model input features (first 5 samples)'.
- Bagging (Ensemble Learning): The notebook pulls individual fitted trees out of the ensemble (bagging_model.estimators_[0] and [49]) and plot_tree's them to reveal how the internal trees differ.

Supporting content in these sessions: 2 reading, 2 in_class_quiz, 2 mcq

---

## Signature analogies

Everyday analogies the course uses to introduce abstract concepts before the formalism — reuse these for on-style content.

- **Apriori Algorithm:** Opens 'Imagine you go to your favorite grocery store... you notice that bread and butter are always kept next to each other' before defining the algorithm.
- **Assumptions of Linear Regression:** 'Think of these assumptions as the rules of a game. If the rules are followed, the game (model) works well' introduces the assumptions before each is defined.
- **Bias and Variance:** Bias/variance made concrete on spam: 'if the model only checks for the word free... demonstrating high bias' vs 'uses every word... capturing irrelevant noise' overfitting.
- **Boosting methods and Adaboost:** 'AdaBoost works like a team of weak models, each learning from the mistakes of the previous one... the next model pays extra attention to those errors' before the formal weight/alpha math.
- **Decision Tree Part 1:** A 20-students 'Play Cricket' splitting scenario (by Height, Performance, Classroom) introduces tree splitting before the formal 'Introduction to DT' and terminology slides.
- **Dimentionality Reduction:** 'Real-World Analogy: choosing a phone using Battery/Camera/Price (easy) vs 200 specifications (confusing)' before the formal 'Curse of Dimensionality' definition.
- **Dimentionality Reduction Part 2:** 'Scenario: Fruit Measurements Problem' (weight, diameter, sweetness; weight and diameter correlated -> redundancy) precedes 'Introducing PCA'.
- **Evaluation Metrics:** 'Imagine you work for a tech company that develops an email filtering system...' scenario opens the deck before the confusion matrix is defined.
- **Feature Engineering - 2:** Feature importance introduced via analogy: 'Imagine a team project where you want to know which member contributed the most to the final output. Similarly, in machine learning, we assess which features are most responsib
- **Feature Importance Techniques:** Cricket-team analogy for LOFO: remove opening batsman/fast bowler/fielder and see the drop; 'In LOFO, features are players, and model performance is the match result' before the formal Working-of-LOFO steps.
- **Gradient Descent Part - 1:** Crumpled-paper-into-dustbin analogy ('He reduces the force slightly and throws again') and foggy-hill analogy precede the cost function and derivatives.
- **Hierarchical Clustering:** 'This is like grouping friends in a room. First, people closest to each other form pairs'; 'A dendrogram is like a family tree for clusters.'
- **Introduction to Ensemble Algorithms:** Blind-men-and-elephant framing ('It's a Spear / It's a Fan / It's a Snake / It's a Tree...') and 'Wisdom of crowd - KBC, product review' precede the formal definitions.
- **KNN:** KNN introduced on a concrete CGPA/IQ -> Placed(Yes/No) student-placement scatter before any distance-metric math is shown.
- **KNN Regression:** A concrete Bill Amount (X) vs Tip (Y) example with a scatter plot precedes the formal 'Model Building Steps'.
- **Logistic Regression:** 'Suppose we need to classify patients as diabetic or non-diabetic based on features like blood test results, BMI, and blood pressure' introduces the algorithm before any equation.
- **Naive Bayes Part 1:** Ramesh/Suresh anonymous-email word-usage scenario (love/wonderful/great weightages) used to introduce the classifier before the Bayes theorem formula.
- **Problem Statement:** Skewness introduced with a concrete scenario 'most customers spend between GBP 100-500, but a few spend GBP 50,000+' before the formal log1p transform is applied.
- **SVM:** SVM intuition built on a concrete CGPA/IQ scatter with candidate lines A/B/C ('identify the one which classifies the best'); the math is explicitly postponed.
- **Simple Linear Regression:** Bill Amount vs Tip data table and scatter plot are shown before the y = mx + c equation is introduced.
- **Support Vector Regression:** The Bill Amount (X) vs Tip (Y) example table and scatter appear in the Introduction before the mathematical intuition.
- **Voting:** 'Quick Intuition: Think of hard voting like a democratic election: Each model casts one vote for a class; the class with the most votes wins.'

## Signature code idioms

Course-specific code patterns that recur across the implementation notebooks. Generate on-style code with these, not generic equivalents.

- **Kaggle-style submission** — Build model, predict on held-out test, emit a named `submission_df` with id + prediction columns — the shape every coding assignment grades.
  ```python
  model.fit(X_train, y_train)
  preds = model.predict(X_test)
  submission_df = pd.DataFrame({'ID': ids, 'target': preds})
  ```
- **Elbow + Silhouette for K** — Never pick K by fiat: sweep K, plot WCSS (elbow) and Silhouette, choose the K they agree on.
  ```python
  for k in range(2, 11):
      km = KMeans(k).fit(X)
      wcss.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
  ```
- **Scale -> PCA -> cluster** — Standardize first, reduce with PCA, then cluster/evaluate on the reduced space — the unsupervised pipeline.
  ```python
  Xs = StandardScaler().fit_transform(X)
  Xp = PCA(n_components=k).fit_transform(Xs)
  labels = KMeans(n).fit_predict(Xp)
  ```
- **groupby -> aggregate -> sort** — Answer 'top/most per group' by grouping, aggregating, then sorting or taking idxmax/idxmin — keeps the entity label.
  ```python
  agg = df.groupby('team')['runs'].sum()
  top = agg.sort_values(ascending=False)
  ```
- **Scale before distance/margin models** — KNN and SVM are scaled explicitly because the algorithm depends on feature magnitude; trees are not.
  ```python
  X_train = StandardScaler().fit_transform(X_train)  # required for KNN / SVM
  ```
- **Leakage-safe split** — Split first, fit the scaler on train only, transform test with it; stratify to preserve class balance.
  ```python
  X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2)
  sc = StandardScaler().fit(X_tr)
  ```
- **Named-estimator ensembles** — Voting/Stacking are built from explicitly named base estimators, then compared to the single-model baseline.
  ```python
  VotingClassifier(estimators=[('dt', dt), ('rf', rf), ('lr', lr)], voting='soft')
  ```

## Frameworks & tools

Libraries used across the implementation notebooks (extracted from imports), grouped by role. Versions are representative 2026 defaults: the notebooks install unpinned, so no exact versions exist in source — replace with your environment's specs. Notebooks = number of implementation notebooks importing the library.

### Core data & numerics

| Library | Version | Notebooks | Used for |
|---|---|---|---|
| `numpy` | 2.1.x | 42 | arrays and vectorized numeric ops |
| `pandas` | 2.2.x | 45 | load, inspect and clean tabular data |
| `warnings` | stdlib | 29 | silence library warnings in notebooks |
| `scipy` | 1.14.x | 8 | distances, stats, hierarchical clustering / dendrograms |
| `time` | stdlib | 2 | time training runs |
| `gdown` | 5.2.x | 2 | download datasets from Google Drive |
| `collections` | stdlib | 1 | Counter / defaultdict for counts |
| `os` | stdlib | 1 | file paths and IO |

### Machine learning

| Library | Version | Notebooks | Used for |
|---|---|---|---|
| `sklearn` | 1.6.x | 38 | estimators, preprocessing, metrics, model selection |
| `xgboost` | 2.1.x | 4 | gradient-boosted decision trees |
| `mlxtend` | 0.23.x | 2 | decision-region plots and stacking helpers |
| `statsmodels` | 0.14.x | 3 | OLS and statistical regression |

### Visualization

| Library | Version | Notebooks | Used for |
|---|---|---|---|
| `matplotlib` | 3.10.x | 44 | plots: loss curves, decision boundaries, scatter |
| `seaborn` | 0.13.x | 40 | heatmaps and statistical plots |
| `plotly` | 5.24.x | 2 | interactive charts |
| `sweetviz` | 2.3.x | 1 | automated EDA reports |

### Other / runtime

| Library | Version | Notebooks | Used for |
|---|---|---|---|
| `google` | Colab runtime | 1 | mount Drive / upload data (google.colab) |
| `itertools` | stdlib | 1 | combinatoric iterators |
| `IPython` | 8.x | 1 | notebook display helpers |

## Glossary

Definition of every concept in the vocabulary the course teaches.

- **AdaBoost** — adaptive boosting - Boosting methods and Adaboost
- **Apriori Algorithm** — candidate-generation association mining - Apriori Algorithm
- **Association Rule Learning** — frequent itemsets and rules - Apriori Algorithm, Eclat
- **Assumptions of Linear Regression** — linearity, independence, etc. - Assumptions of Linear Regression
- **AUC-ROC Curve** — threshold-independent classifier evaluation - AUC - ROC Curve
- **Bagging** — bootstrap aggregation - Bagging
- **Bias and Variance** — under/overfitting tradeoff - Bias and Variance
- **Boosting** — sequential error-correcting ensembles - Boosting methods and Adaboost
- **Choosing K** — selecting number of neighbors - KNN, KNN Implementation
- **Class Imbalance Handling** — class_weight or resampling for skewed classes - implementation notebooks
- **Classification** — predicting discrete labels - Introduction to Machine Learning
- **Clustering** — grouping by similarity - Introduction to Unsupervised Learning
- **Cross-Validation** — k-fold evaluation for tuning - HyperParameter Tuning **(check - confirm it's taught here)**
- **Data Analysis with Pandas** — loading, inspecting, manipulating tabular data - DA with Pandas Part 1 & 2
- **Data Cleaning** — nulls, duplicates, redundant columns - DA with Pandas
- **Data Visualization** — plots to interpret data - DA with Pandas
- **DBSCAN** — density-based clustering - DBSCAN
- **Decision Tree Regression** — trees for continuous targets - Decision Tree Regression
- **Decision Trees** — recursive splitting into decision rules - Decision Tree Part 1 & 2
- **Dimensionality Reduction (PCA)** — reduce features while keeping variance - Dimentionality Reduction (Parts 1-3)
- **Distance Metrics** — distance measures between points - KNN
- **Eclat Algorithm** — vertical-format association mining - Eclat
- **Elbow Method** — choosing number of clusters via the inertia curve - K-Means, DBSCAN implementations
- **Ensemble Learning** — combining multiple models - Introduction to Ensemble Algorithms
- **Euclidean Distance** — straight-line distance - KNN
- **Evaluation Metrics** — accuracy, precision, recall, F1, etc. - Evaluation Metrics
- **Exploratory Data Analysis (EDA)** — summarizing and understanding data - DA with Pandas, capstone EDA sessions
- **Feature Engineering** — creating/transforming features - Feature Engineering 1 & 2 (capstones)
- **Feature Importance** — ranking predictor contribution - Feature Importance Techniques
- **Feature Scaling** — normalizing/standardizing ranges - KNN Advanced
- **Gradient Boosting** — boosting via gradient descent on residuals - Gradient Boosting Classification & Regression
- **Gradient Descent** — iterative optimization of parameters - Gradient Descent Part 1 & 2
- **Hamming Distance** — differences between equal-length strings - KNN
- **Hard vs Soft Margin** — strict vs tolerant separation (parameter C) - SVM
- **Hierarchical Clustering** — nested cluster tree (dendrogram) - Hierarchical Clustering
- **HyperParameter Tuning** — searching for good hyperparameters - HyperParameter Tuning Part 1 & 2
- **Hyperplane and Margin** — separating boundary and its margin - SVM
- **K-Means Clustering** — centroid-based clustering - K-Means Clustering (Parts 1-3)
- **K-Nearest Neighbors (KNN)** — classify by nearest neighbors - KNN, KNN Implementation
- **Kernel Trick** — project to higher dimension for separability - SVM
- **Kernels (Linear / RBF / Polynomial)** — kernel functions - SVM
- **KNN Regression** — regression via nearest neighbors - KNN Regression
- **Label Encoding** — categorical -> integer codes - SVM Implementation
- **Logistic Regression** — linear model for classification - Logistic Regression
- **Machine Learning** — learning from data without explicit programming - Introduction to Machine Learning
- **Machine Learning Lifecycle** — define -> collect -> prepare -> train -> deploy -> monitor - Machine Learning Life Cycle
- **Manhattan Distance** — sum of absolute differences - KNN
- **Minkowski Distance** — generalized distance (parameter p) - KNN
- **Missing Values & Outlier Treatment** — handling gaps and extremes - Missing Values and Outliers Treatment
- **ML Environment Setup** — Anaconda, Jupyter, pip, packages - Setting Up ML Environment
- **Multicollinearity (VIF)** — correlated predictors detected via variance inflation factor - Assumptions of LR, Regularization
- **Multiple Linear Regression** — several predictors - Multiple Linear Regression
- **Naive Bayes** — probabilistic classifier from Bayes' theorem - Naive Bayes Part 1 & 2
- **Normalization (Min-Max Scaling)** — rescale to [0,1] - KNN Advanced
- **One-Hot Encoding** — categorical features to binary indicator columns - implementation notebooks
- **Polynomial Regression** — non-linear polynomial fit - Polynomial Regression
- **Probabilistic KNN** — class probabilities from neighbor proportions - KNN Advanced
- **Random Forest** — bagged decision trees - Random Forest
- **Regression (task)** — predicting continuous values - Introduction to Machine Learning
- **Regression Metrics (R2/MSE/RMSE/MAE)** — error and fit measures for regression - implementation notebooks
- **Regularization (L1/L2)** — penalizing complexity - Regularization
- **Silhouette Score** — cluster cohesion and separation quality metric - clustering implementations
- **Simple Linear Regression** — one predictor, straight-line fit - Simple Linear Regression
- **Stacking** — meta-model over base models - Stacking
- **Standardization (Z-score Scaling)** — mean 0, std 1 - KNN Advanced
- **Supervised Learning** — learns from labeled data - Introduction to Machine Learning
- **Support Vector Machines (SVM)** — maximum-margin classifier - SVM, SVM Implementation
- **Support Vector Regression** — SVM for continuous targets - Support Vector Regression
- **Support Vectors** — points defining the margin - SVM
- **Train-Test Split** — hold-out split for evaluation - SVM Implementation **(check)**
- **Unsupervised Learning** — finds structure in unlabeled data - Introduction to Machine Learning, Introduction to Unsupervised Learning
- **Voting** — combine predictions by vote/average - Voting
- **Weighted KNN** — closer neighbors weigh more - KNN Advanced
- **XGBoost** — regularized gradient boosting - XG boost

## Naming audit

Session and asset names checked for the issues flagged in other KBs (trailing/leading whitespace, double spaces, en/em dashes).

- Session names (103): clean, no issues.
- Asset names (356): double space: 4; embedded newline / concatenated titles: 40; en/em dash: 1.
  - double space: `Bias and Vairance - MCQs`
  - double space: `Boosting Implementation - Classification Boosting Implementation - Classification boosting`
  - double space: `Capstone Project: Ensemble Learning Capstone Project: Ensemble Learning capstone_ensemble_`
  - double space: `Capstone Project -Problem statement [Unsupervised Learning] Capstone Project - Problem Sta`
  - embedded newline / concatenated titles: `Introduction to Ensemble Algorithms Introduction_to_Ensemble_Algorithms_MCQs.zip`
  - embedded newline / concatenated titles: `Voting Voting_MCQs.zip`
  - embedded newline / concatenated titles: `Stacking Stacking_Tutorial_Mcqs_Updated.zip`
  - embedded newline / concatenated titles: `Bagging Bagging_Tutorial_Mcqs.zip`
  - en/em dash: `AUC – ROC Curve AUC_ROC_Curve_MCQs.zip`

## Findings & recommendations

- **Tighten assignment-concept alignment.** Add a one-line 'intended-technique present' check to the four technique-agnostic classification assignments (KNN, Naive Bayes, Logistic, Hyperparameter Tuning); the strong assignments already show the pattern (isinstance SGDRegressor, Ridge+Lasso).
- **Fix two assignment bugs:** KNN_SVM_DT `svm_mse > 1` gates nothing; SVM tells students ≥0.82 but grades ≥0.80.
- **Add Entropy / Information Gain** to the concept vocabulary — taught in Decision Tree but untracked.
- **Calibrate soft thresholds** (KNN/Logistic 65%, K-Means Silhouette 0.25) against reference solutions.
- **Readings/quizzes/MCQs are cataloged but unread** — the next content source to ingest if the KB should reason over them, not just count them.
- **Visual pedagogy is uncaptured** (text-only extraction) — a slide-image pass would add the diagram/progressive-reveal patterns the theory sessions rely on.

## Eval set

Checkable pass/fail rules for judging new Classical ML content against the course's house style and
its known defects. Follows the standard eval-set pattern: each rule carries a `key` (the pedagogy
pattern or gap it enforces), a `type` (pedagogy vs content-accuracy), an `applies_to` scope, a
`check` question, `pass` criteria, grounded `topic_examples` (illustrative passes), and — on the subtle
rules — a `fail_example` (a near-miss to reject).

Run every generated session, notebook, MCQ set, or coding assignment through the applicable items
before shipping; report each failure with the offending slide/section/cell.

```yaml
eval_set:
  version: 3
  course: classical-ml
  scope_note: >
    Defines the terms used in `applies_to` so conditional rules fire only on matching artifacts.
    session_type: concept (theory/slides) | implementation (notebook) | capstone.
    chain_topic: a topic taught as an ordered fix-the-flaw sequence (e.g. simple->regularized
      regression, bagging->boosting).
    algorithm_topic: any topic introducing a model family (KNN, SVM, NB, trees, regression,
      clustering, ensembles, PCA, association rules).
    math_topic: a topic with a real derivation/formula (gradient descent, SVM margin, PCA,
      evaluation metrics, association-rule metrics).
    implementation_session: a notebook-backed session.
    coding_assignment: an IDE coding assignment (not a taught notebook).
    non_opening: any session that is not the first in its topic.
    Items with `applies_to: all` are universal; apply the rest only when the artifact matches.
    topic_examples are illustrative, NOT definitional — "does not match the example" is never itself
    a fail; a valid variation on a different domain passes.

  items:
    # ---------- pedagogy / house-style gates ----------
    - id: CML01
      key: agenda_first_key_takeaways_last
      type: pedagogy
      applies_to: all
      check: "Opens with an Agenda listing today's concepts and closes with Key Takeaways that mirror it?"
      pass:  "Both present; takeaways restate the agenda items, no new content."
      topic_examples:
        knn: "Agenda (what is KNN, distance metrics, choosing K) mirrored by Key Takeaways."
        evaluation_metrics: "Agenda (confusion matrix, accuracy, precision/recall) mirrored at close."

    - id: CML02
      key: real_world_motivation_named_domains
      type: pedagogy
      applies_to: all
      check: "Is the concept anchored in a specific named domain (spam, fraud, churn, medical, retail) early and carried through the examples?"
      pass:  "A concrete named domain appears up front and recurs; not abstract 'some data'."
      topic_examples:
        svm: "Mushroom classification (cap, gills, stalk...) carried through the SVM notebook."
        kmeans: "Mall customer segmentation carried through the whole K-Means notebook."
      fail_example: "Opens with 'consider a dataset X with n features' and never names a domain."

    - id: CML03
      key: analogy_before_formalism
      type: pedagogy
      applies_to: concept_sessions
      check: "Does an abstract/statistical concept get an everyday analogy before the math, while mechanical ops (fit/predict, indexing) get none?"
      pass:  "Analogy precedes formalism for conceptual topics; no forced analogy on mechanical steps."
      topic_examples:
        gradient_descent: "Foggy-hill / crumpled-paper analogy before the cost function and derivatives."
        hierarchical_clustering: "'A dendrogram is a family tree for clusters' before the algorithm."
      fail_example: "Bolting a laboured analogy onto train_test_split or .fit() (a mechanical op that needs none)."

    - id: CML04
      key: build_by_fixing_previous_limitation
      type: pedagogy
      applies_to: chain_topics
      check: "If the method improves on a prior one, does it first name the exact limitation it fixes?"
      pass:  "The prior method's specific shortcoming is stated as the motivation; methods are ordered, not an unordered list."
      topic_examples:
        regression: "Simple LR -> its limits -> Multiple/Polynomial -> overfitting -> Regularization."
        knn_to_svm: "KNN's boundary limits motivate SVM's max-margin separation."
      fail_example: "Introducing Regularization as just 'another technique' without naming the overfitting it fixes."

    - id: CML05
      key: comparison_table_contrasting_methods
      type: pedagogy
      applies_to: algorithm_topics
      check: "Is a new algorithm placed beside a sibling across shared dimensions (assumptions, speed, output), used to say when to prefer each?"
      pass:  "A side-by-side contrast is present and drives a when-to-use statement."
      topic_examples:
        clustering: "Hierarchical vs K-Means table (approach, #clusters, output, flexibility)."
        dimensionality: "Feature Selection vs Feature Extraction table."

    - id: CML06
      key: when_it_fails_honesty
      type: pedagogy
      applies_to: all
      check: "Are the method's failure modes / assumptions stated alongside its strengths, and is any misleading metric shown concretely?"
      pass:  "Limitations stated; a misleading case shown with real numbers where relevant."
      topic_examples:
        kmeans: "Cons slide: difficulty choosing K, sensitive to init/scaling, struggles with non-spherical clusters."
        evaluation_metrics: "Accuracy paradox: 98% accuracy shown as misleading on imbalanced data."
      fail_example: "Presenting a model with only pros ('SVM is powerful and accurate') and no failure modes."

    - id: CML07
      key: worked_example_step_by_step
      type: pedagogy
      applies_to: math_topics
      check: "Is at least one formula taught by plugging concrete numbers through explicit steps to a decision?"
      pass:  "A small numeric worked example is present, not left purely symbolic."
      topic_examples:
        knn: "A concrete CGPA/IQ -> Placed scatter with a worked nearest-neighbour vote."
        naive_bayes: "Ramesh/Suresh word-usage scenario computed to a class decision."
      fail_example: "Showing Bayes' theorem symbolically and moving on without a single plugged-in number."

    - id: CML08
      key: ml_workflow_skeleton
      type: pedagogy
      applies_to: implementation_session
      check: "Does the notebook follow load -> inspect/EDA -> clean/preprocess -> train -> evaluate with recognizable section headings?"
      pass:  "The pipeline order and headings are present and consistent."
      topic_examples:
        svm: "Define problem -> Load -> EDA -> Train-test split -> Build SVM -> Evaluate."
        kmeans: "Import -> Load -> Inspect -> Clean -> Scale -> Model -> Elbow -> Silhouette -> Interpret."

    - id: CML09
      key: observe_and_interpret_loop
      type: pedagogy
      applies_to: implementation_session
      check: "Is every plot or computed output followed by a one-to-three sentence interpretation phrased as an observation/decision?"
      pass:  "Numbers are converted into meaning after each output; not code-only narration."
      topic_examples:
        kmeans: "'WCSS drops sharply to K=5... Hence K=5' after the elbow plot."
        missing_values: "'360 repeats most in loan term, so impute with the mode' after the value counts."
      fail_example: "A notebook that plots a chart and moves straight to the next cell with no written takeaway."

    - id: CML10
      key: preprocessing_justified_by_algorithm
      type: pedagogy
      applies_to: algorithm_topics
      check: "Is each preprocessing step justified by the algorithm's need (scale for KNN/SVM; no scaling for trees; encode categoricals)?"
      pass:  "The reason for each step ties to a model property, not applied as boilerplate."
      topic_examples:
        knn: "Feature scaling justified because KNN uses distance, so magnitudes must be comparable."
        svm: "Scaling stated as required before SVM; label encoding to make categoricals model-consumable."
      fail_example: "StandardScaler applied before a Decision Tree 'because we always scale' (trees don't need it)."

    - id: CML11
      key: leakage_safe_split
      type: pedagogy
      applies_to: implementation_session
      check: "Is the split done first, the scaler fit on train only, and the split stratified where classes are imbalanced?"
      pass:  "No fit on the full dataset before splitting; test set untouched; stratify present for classification."
      topic_examples:
        hyperparameter_tuning: "Train-test split with stratification before tuning; scaler fit on train."
      fail_example: "StandardScaler().fit_transform(X) on the whole dataset, THEN train_test_split (leakage)."

    # ---------- content-accuracy gates ----------
    - id: CML12
      key: metric_defined_before_use
      type: content-accuracy
      applies_to: math_topics
      targets: metric_used_before_defined
      check: "Is every metric (accuracy/precision/recall/F1/R2/RMSE/Silhouette/support/confidence/lift) defined before it is computed or plotted?"
      pass:  "Definition precedes use; the computation is shown, not assumed to pre-exist."
      topic_examples:
        evaluation_metrics: "Confusion matrix defined before accuracy/precision/recall are computed."
        association_rules: "Support/confidence/lift defined before rules are filtered by them."
      fail_example: "Heat-mapping .corr() output without ever defining correlation."

    - id: CML13
      key: regularization_l1_l2_distinction
      type: content-accuracy
      applies_to: regularization_sessions
      check: "Are L1 and L2 kept distinct (L1 drives weights to exactly zero -> sparsity; L2 shrinks, none exactly zero), never conflated?"
      pass:  "L1=Lasso=feature selection; L2=Ridge=correlated-feature shrink; no claim that L2 zeroes weights."
      topic_examples:
        regularization: "L1 zeroes coefficients (feature selection); L2 shrinks smoothly for correlated inputs."
      fail_example: "Saying 'L2 removes unimportant features by setting weights to zero' (that is L1)."

    - id: CML14
      key: clustering_k_selection
      type: content-accuracy
      applies_to: clustering_sessions
      check: "Is K chosen via elbow + silhouette (not by fiat), and are the algorithm's assumptions/limits stated?"
      pass:  "A K-selection method is shown; K-Means assumptions (spherical, scale-sensitive) or DBSCAN eps/min_samples stated."
      topic_examples:
        kmeans: "K swept 2..10, elbow + silhouette agree on K; assumptions (spherical clusters, scaling) named."
        dbscan: "eps/min_samples tuned; silhouette computed excluding noise."
      fail_example: "'We choose K=5' with no elbow/silhouette and no reason."

    - id: CML15
      key: distance_margin_scaling_and_metric
      type: content-accuracy
      applies_to: distance_model_sessions   # KNN, SVM
      check: "For distance/margin models, is scaling stated as required, and (KNN) the distance-metric choice explained?"
      pass:  "Scaling named as required for KNN/SVM; distance metric (Euclidean/Manhattan/Minkowski/Hamming) chosen by feature type."
      topic_examples:
        knn: "Euclidean vs Manhattan vs Hamming by feature type; K odd to avoid ties."
      fail_example: "Training KNN on raw unscaled features without noting distance is scale-sensitive."

    - id: CML16
      key: tree_split_criterion
      type: content-accuracy
      applies_to: tree_sessions
      check: "Are decision-tree splits explained via impurity (entropy / information gain or Gini), and is it stated trees need no feature scaling?"
      pass:  "Split criterion tied to node purity; no scaling requirement claimed for trees."
      topic_examples:
        decision_tree: "Entropy/information gain drives split choice; purer child nodes preferred."
      fail_example: "Explaining a decision tree without any notion of impurity/information gain."

    - id: CML17
      key: concept_vocabulary_and_depth
      type: content-accuracy
      applies_to: all
      targets: untracked_concept
      check: "Are the concepts taught part of the course vocabulary (or explicitly added), and are outcomes tagged at an appropriate coverage depth (Introduced/Explained/Applied/Deep-dive)?"
      pass:  "No concept taught as new when untracked (e.g. Entropy/Information Gain must be added to the vocabulary); outcomes carry a depth tag."
      topic_examples:
        decision_tree: "Entropy/Information Gain is taught but not yet in the concept list -> add it, don't leave it untracked."
```


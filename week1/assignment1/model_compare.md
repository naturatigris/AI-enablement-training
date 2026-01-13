# AI Model Evaluation Sheet

**Department Applications:**  
- Application Development (Code Creation & Quality)  
- Data Analysis & SQL Generation  
- Infrastructure Automation & DevOps  
- Usability, Responsiveness, and Latency

**Models Reviewed:**  
- GPT-5  
- Claude 4.5  
- Gemini Flash  
- DeepSeek-R1:7B (Ollama)  

**Rating Key:**  
- **excellent** – highly capable, production-ready  
- **good** – reliable, minor gaps  
- **basic / limited support** – works partially, may need fixes  
- **not supported** – unsuitable or fails  

---

## AI Model Evaluation Table

| Use Case / Criteria                        | GPT-5           | Claude 4.5    | Gemini Flash    | DeepSeek-R1:7B (Ollama) | Notes |
|--------------------------------------------|-----------------|-----------------|----------------|-------------------------|-------|
| **AppDev – Code Creation & Quality** | excellent       | good            | good           | not supported           | **Prompt:** "code for implementing user registration and login"<br>**GPT-5:** Backend API with Node.js + Express, structured controllers/routes/store, bcrypt password encryption, JWT-based authentication, proper HTTP responses; minor issue: no centralized error middleware, secrets could be environment-based.<br>**Claude:** Frontend-centric HTML/CSS/JS solution storing credentials in localStorage; visually well-designed but lacks real security, plain-text passwords, no server verification.<br>**Gemini:** PHP + MySQL implementation with prepared statements, secure password hashing, session management; safe but less modular and verbose.<br>**DeepSeek:** Output inconsistent, some syntax errors, incomplete or non-functional code, unsuitable for deployment. |
| **Data – SQL Generation**                   | excellent       | excellent       | good           | basic / limited support | **Prompt:** "give me an sql query to calculate the second oldest person in the list."<br>claude , chatgpt,gemini were able to produce correct result.DeepSeek-R1:7B model produced invalid sql statement and the aulity of the output generated was extremely bad as it confused the concept between ranking and aggregates resulting in an output with exposed chain of thoughts reasoning that can be misleaing. |
| **Infra Automation – DevOps Scripts** | excellent       | excellent       | good           | not supported           | **Prompt:** "terraform script to provision an S3 bucket and upload files"<br>GPT-5: Complete, production-ready, error-free.<br>Claude 4.5: Functional script with extra validation and explanatory comments, slightly verbose.<br>Gemini Flash: Minimal working script, meets basic requirements, limited error handling.<br>DeepSeek-R1:7B: Logic errors, invalid syntax, would fail in real deployments. |
| **Ease of Use**                             | excellent       | excellent       | good           | basic / limited support | GPT-5: Minimal guidance required, consistent output.<br>Claude: Step-by-step explanations, clear but slightly verbose.<br>Gemini Flash: Fast and simple.<br>DeepSeek-R1:7B: Needs careful prompt design, inconsistent outputs. |
| **Speed / Latency**                         | excellent       | good            | excellent      | good                    | GPT-5: Fast and stable.<br>Claude: Slightly slower due to verbose outputs, but accurate.<br>Gemini Flash: Very fast, ideal for time-sensitive tasks.<br>DeepSeek-R1:7B: Variable speed depending on task complexity, slower for reasoning-heavy prompts. |

---

## Summary / Recommendations

- **Best All-Round Choice:** GPT-5  
- **Best for Learning / Reports / Explanations:** Claude 4.5  
- **Best for Fast, Lightweight Tasks:** Gemini Flash  
- **Experimental / Local Testing:** DeepSeek-R1:7B (Ollama)  

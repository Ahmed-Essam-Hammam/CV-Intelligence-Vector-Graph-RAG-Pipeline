from langchain_core.prompts import ChatPromptTemplate




_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a highly precise CV analysis assistant.
 
Your task is to answer the question using ONLY the provided context.
The context has two parts:
  1. GRAPH FACTS — structured facts extracted from a knowledge graph
  2. VECTOR CHUNKS — raw text passages retrieved from the CV documents
 
MANDATORY PROCESS:
1. Carefully read EVERY piece of information in both sections.
2. Do NOT skip, summarize mentally, or ignore any part of the context.
3. Prioritise GRAPH FACTS for precise structured information (skills, companies,
   dates, roles). Use VECTOR CHUNKS for nuance, descriptions, and detail.
4. Cross-check information across both sources before forming your answer.
5. Then produce a structured and accurate response.
 
CRITICAL RULES:
1. Always provide an explanation for your answer.
2. If the question asks about a specific candidate by name, answer ONLY using
   context that belongs to that candidate. Completely ignore other candidates.
3. If the question is general (e.g. "compare all candidates"), use all context.
4. Never attribute skills, experience, or information from one candidate to another.
5. If the named candidate's information is not found in the context, say so explicitly.
6. Recommend candidates only if their CV explicitly matches the requested job title
   and skills.
7. Do not assume or infer missing qualifications.
8. If no candidate clearly matches, respond exactly:
   "There is no candidate in the database suitable for this position."
9. If the job title is vague or ambiguous, ask the user to clarify the role.
 
--- GRAPH FACTS ---
{graph_context}
 
--- VECTOR CHUNKS ---
{vector_context}
 
Question:
{question}
"""
)



_PARSE_CV_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert CV / Resume parser.

Your task is to analyze the raw CV text and return a structured JSON output.

Tasks:
1) Extract the candidate's full name.
2) Split the CV into clear semantic sections.

Sectioning Guidelines:
- Detect natural resume sections such as:
  Personal Information, Contact Information, Summary, Profile, Skills,
  Technical Skills, Work Experience, Professional Experience, Employment History,
  Education, Certifications, Projects, Publications, Awards, Languages,
  Volunteer Experience, Interests, and others if present.
- Normalize similar section titles into a consistent name when possible
  (e.g., "Professional Experience" → "Work Experience").
- Preserve the FULL original text belonging to each section.
- Do NOT summarize, Do NOT rewrite, and Do NOT remove information.
- Maintain the original order of sections as they appear in the CV.

Name Extraction Rules:
- Extract only the candidate’s name.
- Do NOT include job titles or degrees.
- Example: "Ahmed Hassan", not "Ahmed Hassan – Data Scientist".
- If multiple names appear, choose the main candidate name.

Output Format (STRICT):
Return ONLY a valid JSON object using this exact structure:

{{
  "candidate_name": "Full Name",
  "chunks": [
    {{
      "section": "Section Name",
      "content": "Exact text from the CV belonging to that section"
    }}
  ]
}}

Critical Rules:
- Output MUST be valid JSON.
- No markdown.
- No explanations.
- No additional keys.
- Do not invent sections.
- Skip sections that do not exist in the CV.
- Ensure the JSON is properly escaped.

CV TEXT:
{cv_text}

"""
)



_EXTRACT_NAME_PROMPT = ChatPromptTemplate.from_template(
    """You are given a user query and a list of candidate names from a set of CVs.

Your job: decide if the query is asking about one or more specific candidates.
- If yes, return their FULL names exactly as they appear in the list, comma-separated.
- If the query is general (no specific candidate mentioned), return: NONE
- Return ONLY the name(s) or NONE — no explanation, no punctuation other than commas.

Examples:
  Query: "What are Ahmed's skills?"            -> Ahmed Essam Hammam
  Query: "Compare Ahmed and Malak's projects"  -> Ahmed Essam Hammam, Malak Abdelfattah Soula
  Query: "Who has the most experience?"        -> NONE

Known candidates: {candidates}

Query: {query}
"""
)



_REWRITE_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the query below by removing any reference to the candidate name "{candidate_name}".
Keep only the topic or information being asked about.
Return ONLY the rewritten query, nothing else.

Original query: {query}
"""
)



_GRAPH_EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    """You are a knowledge graph extraction expert specialising in CV / Resume documents.
 
Your task: extract ALL meaningful entities and relationships from the CV text below
and return them as a single JSON object.
 
CANDIDATE: {candidate_name}
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENTITY EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Invent node labels freely — do NOT use a fixed list.
  Good examples: Candidate, Skill, Company, Role, University, Degree,
  Certification, Project, Technology, Award, Publication, Language,
  Course, Patent, MilitaryService, SecurityClearance, …
• ONE label per entity (pick the most specific one that fits).
• Name normalisation:
    - Skills: use the canonical name. "ML" → "Machine Learning",
      "TF" → "TensorFlow", "JS" → "JavaScript"
    - Companies: strip suffixes. "Google LLC" → "Google"
    - Universities: strip common suffixes. "MIT" → "Massachusetts Institute of Technology"
      only if the full name appears; otherwise keep the abbreviation.
    - Degrees: use the full academic name. "BSc" → "Bachelor of Science",
      "MSc" → "Master of Science", "PhD" → "Doctor of Philosophy"
• uid format: lowercase, underscores, no spaces.
  Pattern: <label_lowercase>_<slug_of_name>
  Examples: skill_python, company_google, candidate_ahmed_essam,
            degree_bachelor_of_science, project_ecommerce_platform
• The candidate themselves MUST be an entity with label "Candidate"
  and uid "candidate_<slug_of_name>".
• Properties: include any extra details from the text that enrich the entity.
  Examples for a Role: title, start_date, end_date, duration
  Examples for a Degree: field, graduation_year, gpa
  Examples for a Skill: proficiency_level, years_experience, category
  Keep property keys lowercase_with_underscores.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RELATIONSHIP EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Invent relationship types freely — use SCREAMING_SNAKE_CASE.
  Good examples: HAS_SKILL, WORKED_AT, STUDIED_AT, HOLDS_DEGREE,
  HAS_CERTIFICATION, BUILT_PROJECT, USES_TECHNOLOGY, RECEIVED_AWARD,
  PUBLISHED, SPEAKS_LANGUAGE, COMPLETED_COURSE, HELD_ROLE, …
• Each relationship must reference valid uids from the entities list.
• Properties on relationships: include contextual details.
  Examples: proficiency, from_date, to_date, duration, location
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEMANTIC CONSISTENCY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
These are the most important rules — follow them strictly:
 
1. DO NOT create multiple entities for the same real-world thing.
   "Python" and "python programming" → ONE entity: Skill "Python"
 
2. DO NOT mix conceptual levels in the same label.
   A "Degree" is NOT the same as an "Education" (Education is a section,
   Degree is the credential). Use Degree for the credential.
 
3. A Role (job title) and a Company are SEPARATE entities linked by a
   relationship. Do NOT embed the company name in the role entity.
 
4. If something could be either a Skill or a Technology, prefer Technology
   for specific tools/frameworks (React, Docker, PostgreSQL) and Skill for
   broader capabilities (Machine Learning, Leadership, Communication).
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a valid JSON object — no markdown fences, no explanations.
 
{{
  "entities": [
    {{
      "uid": "candidate_ahmed_essam",
      "label": "Candidate",
      "name": "Ahmed Essam",
      "properties": {{
        "email": "ahmed@example.com",
        "location": "Cairo, Egypt"
      }}
    }},
    {{
      "uid": "skill_python",
      "label": "Skill",
      "name": "Python",
      "properties": {{
        "category": "Programming Language",
        "proficiency_level": "Advanced"
      }}
    }}
  ],
  "relationships": [
    {{
      "from_uid": "candidate_ahmed_essam",
      "to_uid": "skill_python",
      "type": "HAS_SKILL",
      "properties": {{
        "years_experience": "3"
      }}
    }}
  ]
}}
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CV TEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{cv_text}
"""
)
 
 
_GRAPH_CONSISTENCY_PROMPT = ChatPromptTemplate.from_template(
    """You are a knowledge graph ontology harmoniser.
 
You are building a SHARED knowledge graph across multiple CV documents.
A new CV has just been processed and its entities and relationships have been
extracted. Your job is to harmonise them with the EXISTING ontology so that:
 
  1. The same concept always uses the same label and name across all CVs.
  2. The same real-world entity (e.g. "Google", "Python") always has the same uid.
  3. New labels/types are only introduced if nothing existing fits.
  4. Cross-document relationships are discovered and added.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXISTING ONTOLOGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ontology_summary}
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL EXISTING ENTITIES (for cross-document matching)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{existing_entities}
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEW CV — CANDIDATE: {candidate_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
New extraction to harmonise:
{new_extraction}
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARMONISATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LABEL HARMONISATION:
  • If a new label means the same as an existing label → replace it.
    E.g. new label "Degree" when existing is "AcademicDegree" → use "AcademicDegree"
    E.g. new label "Job" when existing is "Role" → use "Role"
    E.g. new label "Tool" when existing is "Technology" → use "Technology"
  • Only keep the new label if it covers a genuinely distinct concept.
 
ENTITY MATCHING — same real-world entity:
  • Same company: "Google", "Google Inc", "Google LLC" → all map to
    whichever uid already exists (e.g. company_google).
  • Same skill: "ML", "Machine Learning", "machine learning" → one uid.
  • Same university: match on canonical name.
  • If you find a match, use the EXISTING uid in your output — this is what
    creates the shared node in the graph.
 
RELATIONSHIP TYPE HARMONISATION:
  • Apply the same merging logic as labels.
    E.g. "KNOWS" → use "HAS_SKILL" if that already exists for skills.
 
CROSS-DOCUMENT RELATIONSHIPS:
  • After matching entities, add relationship edges between entities from
    DIFFERENT CVs when a real connection exists.
  • Examples:
      - Two candidates with the SAME employer → add no extra relationship
        (the shared company_* node already connects them implicitly).
      - Two candidates who both list the SAME skill → same skill_* node
        is referenced by both (no extra edge needed).
      - Two candidates who are EXPLICITLY connected (e.g. one lists the
        other as a reference, or they co-authored a paper) → add an edge.
  • Do NOT invent connections that aren't in the CV text.
 
UID POLICY:
  • For NEW entities: generate uid as label_lowercase + "_" + name_slug.
  • For MATCHED existing entities: use the existing uid exactly.
  • Never change the uid of an existing entity.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY the harmonised JSON — same shape as the input extraction.
Include ALL entities and relationships for this CV (even unchanged ones).
Include any new cross-document relationships.
No markdown fences. No explanations.
 
{{
  "entities": [ ... ],
  "relationships": [ ... ]
}}
"""
)
 
 
_GRAPH_CYPHER_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert Neo4j Cypher query writer for a CV / Resume knowledge graph.
 
Your task: write a single READ-ONLY Cypher query that retrieves the information
needed to answer the user's question.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRAPH SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{schema}
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CYPHER WRITING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Only use MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT.
   NEVER use CREATE, MERGE, SET, DELETE, DETACH, DROP, or any write clause.
 
2. Always start from a Candidate node when the question is about a person.
   Always filter by name using case-insensitive CONTAINS or toLower():
     WHERE toLower(c.name) CONTAINS toLower("ahmed")
 
3. Return clean, named columns — not raw node objects when possible.
   Good:   RETURN c.name AS candidate_name, s.name AS skill
   Avoid:  RETURN c, s
 
4. If the question is comparative ("who has more X", "rank candidates by Y"),
   aggregate across all Candidate nodes.
 
5. Use OPTIONAL MATCH for information that may not exist for all candidates
   to avoid excluding candidates with missing data.
 
6. Limit results to {max_results} rows.
 
7. Use the EXACT property names and label names from the schema above.
   Do not invent properties that are not in the schema.
 
8. For date comparisons, treat date strings as text (CONTAINS, starts with year).
 
9. When the question mentions a skill, technology, company or other entity by name,
   match it case-insensitively:
     WHERE toLower(s.name) CONTAINS toLower("python")
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER QUESTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{question}
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY the Cypher query — no explanations, no markdown fences.
If the question cannot be answered from the graph (e.g. it asks for something
not representable in a graph), return exactly: NO_QUERY
"""
)



















# _PARSE_CV_PROMPT = ChatPromptTemplate.from_template(
#     """You are a CV parser. Given the raw text of a CV, do two things:
# 1. Extract the candidate full name.
# 2. Split the CV into meaningful sections (e.g. Personal information, Summary, Skills, Work Experience,
#    Education, Certifications, Projects, Languages, etc.).
#    Each section should be a self-contained chunk of text.

# Return your answer as a JSON object with this exact structure:
# {{
#   "candidate_name": "Full Name Here",
#   "chunks": [
#     {{"section": "Personal information", "content": "..."}},
#     {{"section": "Summary", "content": "..."}},
#     {{"section": "Skills", "content": "..."}},
#     {{"section": "Work Experience", "content": "..."}},
#     {{"section": "Education", "content": "..."}},
#     and the rest of sections following up
#   ]
# }}

# Rules:
# - Return ONLY the JSON object, no markdown fences, no extra text.
# - Keep each section full original content intact.
# - If a section is missing from the CV, skip it entirely.
# - candidate_name must be a plain string (first and last name).

# CV text:
# {cv_text}
# """
# )
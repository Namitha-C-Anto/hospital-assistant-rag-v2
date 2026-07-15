# Total: 24 test cases across 3 PDFs

TEST_DATA = [

    # ─────────────────────────────────────────────
    # admission_and_discharge_guidelines_2003.pdf
    # ─────────────────────────────────────────────

    {
        "id": "Q001",
        "question": "What are the two types of hospital admissions described in the guidelines?",
        "ground_truth": "The guidelines describe two types of hospital admissions: elective admissions, which are planned, and emergency admissions, which are unplanned and require urgent care.",
        "source_doc": "admission_and_discharge_guidelines_2003.pdf",
        "notes": "Basic document understanding."
    },

    {
        "id": "Q002",
        "question": "What are the five strategic objectives of an effective admissions and discharge policy?",
        "ground_truth": "The objectives are to provide integrated patient-centred care, maximize clinical and organizational effectiveness, establish integrated acute care networks, ensure local access with high-quality care, and acquire admissions data for planning and monitoring.",
        "source_doc": "admission_and_discharge_guidelines_2003.pdf",
        "notes": "Multi-point extraction."
    },
    {
        "id": "Q002",
        "question": "According to the guidelines, what are the responsibilities of the Hospital Bed Manager (HBM)?",
        "ground_truth": "The Hospital Bed Manager should oversee bed management, implement the bed management policy, coordinate the bed management team, operate the service continuously (24/7), report to senior management, continuously analyse bed usage and provide reports and forecasts, centralise bed allocation, and ensure patients are accommodated in the most appropriate beds.",
        "source_doc": "admission_and_discharge_guidelines_2003.pdf",
        "notes": "Uses the acronym HBM and the exact role title, making it a strong BM25 test case."
    },

    {
        "id": "Q003",
        "question": "What hospital bed occupancy level is considered desirable for balancing emergency and elective admissions?",
        "ground_truth": "A hospital bed occupancy level of about 85% is considered desirable.",
        "source_doc": "admission_and_discharge_guidelines_2003.pdf",
        "notes": "Numeric retrieval."
    },

    {
        "id": "Q004",
        "question": "How is an emergency hospital admission defined?",
        "ground_truth": "An emergency hospital admission is an unplanned admission resulting from trauma or acute illness that cannot be treated on an outpatient basis.",
        "source_doc": "admission_and_discharge_guidelines_2003.pdf",
        "notes": "Definition retrieval."
    },

    {
        "id": "Q005",
        "question": "According to the guidelines, when should discharge planning begin?",
        "ground_truth": "Discharge planning should begin during the pre-admission stage and continue from the day of admission.",
        "source_doc": "admission_and_discharge_guidelines_2003.pdf",
        "notes": "Checks temporal reasoning."
    },

    {
        "id": "Q006",
        "question": "What information should patients receive during the pre-admission visit?",
        "ground_truth": "Patients and their carers should be informed about the patient's medical condition, proposed treatment, and likely hospital procedures.",
        "source_doc": "admission_and_discharge_guidelines_2003.pdf",
        "notes": "Patient communication."
    },

    {
        "id": "Q007",
        "question": "Into which five categories should Emergency Department patients be streamed?",
        "ground_truth": "Patients should be streamed into resuscitation, minor illness and injury, paediatric cases, specialised medical or surgical assessment, and psychiatric assessment.",
        "source_doc": "admission_and_discharge_guidelines_2003.pdf",
        "notes": "List extraction."
    },

    {
        "id": "Q008",
        "question": "What are the core principles of effective discharge planning?",
        "ground_truth": "Hospital bed use and discharge should be planned before admission where possible, the estimated discharge date should be documented within 24 hours, discharge should be streamlined, and complex discharges should be discussed in multidisciplinary forums.",
        "source_doc": "admission_and_discharge_guidelines_2003.pdf",
        "notes": "Multi-sentence retrieval."
    },

    {
        "id": "Q009",
        "question": "Who should coordinate a patient's discharge plan?",
        "ground_truth": "A nominated member of the multidisciplinary team should coordinate the patient's discharge plan.",
        "source_doc": "admission_and_discharge_guidelines_2003.pdf",
        "notes": "Role identification."
    },

    {
        "id": "Q010",
        "question": "What responsibilities are assigned to the Hospital Bed Manager?",
        "ground_truth": "The Hospital Bed Manager oversees bed management, coordinates bed allocation, manages hospital bed resources, monitors bed utilization, and reports to senior management.",
        "source_doc": "admission_and_discharge_guidelines_2003.pdf",
        "notes": "Role-based retrieval."
    },

    # ─────────────────────────────────────────────
    # doctor_consultation_booking.pdf
    # ─────────────────────────────────────────────

    {
        "id": "Q011",
        "question": "What is the primary purpose of the online doctor's appointment system?",
        "ground_truth": "The system is designed to allow patients to book doctor appointments online, manage appointments, store medical records, and simplify communication between patients and doctors.",
        "source_doc": "doctor_consultation_booking.pdf",
        "notes": "Main objective."
    },

    {
        "id": "Q012",
        "question": "What software development methodology is used to build the application?",
        "ground_truth": "The application is developed using the Waterfall Development Methodology.",
        "source_doc": "doctor_consultation_booking.pdf",
        "notes": "Simple factual retrieval."
    },

    {
        "id": "Q013",
        "question": "What are the three tiers of the application's architecture?",
        "ground_truth": "The application consists of the Presentation Tier, Application Tier, and Data Tier.",
        "source_doc": "doctor_consultation_booking.pdf",
        "notes": "Architecture retrieval."
    },

    {
        "id": "Q014",
        "question": "What can a patient do after successfully logging into the system?",
        "ground_truth": "A patient can select a doctor, book appointments, cancel appointments, view medical history, communicate with doctors, download prescriptions, and log out.",
        "source_doc": "doctor_consultation_booking.pdf",
        "notes": "Functional capabilities."
    },

    {
        "id": "Q017",
        "question": "What information is stored in the database after each patient visit?",
        "ground_truth": "The patient's medical history and appointment-related information are stored in the database for future reference.",
        "source_doc": "doctor_consultation_booking.pdf",
        "notes": "Database understanding."
    },

    {
        "id": "Q018",
        "question": "What features are available on the consultation dashboard?",
        "ground_truth": "The dashboard provides video calling, messaging, prescription download, and storage of consultation records.",
        "source_doc": "doctor_consultation_booking.pdf",
        "notes": "UI feature extraction."
    },

    # ─────────────────────────────────────────────
    # hospital_collection.pdf
    # ─────────────────────────────────────────────

    {
        "id": "Q019",
        "question": "What is the primary objective of the hospital billing and collections policy?",
        "ground_truth": "The objective is to bill patients and payers accurately and promptly while ensuring quality customer service, timely follow-up, compliance with regulations, and consideration of financial assistance.",
        "source_doc": "hospital_collection.pdf",
        "notes": "Policy objective."
    },

    {
        "id": "Q020",
        "question": "What are Extraordinary Collection Actions (ECAs)?",
        "ground_truth": "ECAs are collection activities that can only be taken after reasonable efforts have been made to determine whether a patient qualifies for financial assistance.",
        "source_doc": "hospital_collection.pdf",
        "notes": "Definition retrieval."
    },

    {
        "id": "Q021",
        "question": "How many billing statements are mailed to insured patients before collection actions may be considered?",
        "ground_truth": "Major Hospital mails at least three separate billing statements to insured patients.",
        "source_doc": "hospital_collection.pdf",
        "notes": "Numeric retrieval."
    },

    {
        "id": "Q022",
        "question": "What prompt pay discount is available and under what condition?",
        "ground_truth": "Patients receive a 20% discount if the balance is paid in full within 21 days of the first billing statement.",
        "source_doc": "hospital_collection.pdf",
        "notes": "Checks percentage and timeline."
    },

    {
        "id": "Q023",
        "question": "What processing fee applies to an extended payment plan?",
        "ground_truth": "Extended payment plans that last between 4 and 18 months incur a processing fee of $3.00 per month.",
        "source_doc": "hospital_collection.pdf",
        "notes": "Financial detail retrieval."
    },
]
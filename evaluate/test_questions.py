
# Total: 23 test cases across 5 hospital PDFs

TEST_DATA = [

    # ─────────────────────────────────────────────
    # admission_and_discharge_process.pdf
    # ─────────────────────────────────────────────
    # {
    #     "id": "Q001",
    #     "question": "What are the four main requirements for a patient to get admitted for hospitalization?",
    #     "ground_truth": "Patients requiring hospitalization must obtain a physician recommendation, complete the admission form, submit ID proof, and complete payment or insurance verification.",
    #     "source_doc": "admission_and_discharge_process.pdf",
    #     "notes": "Verifies basic step-by-step admission requirements sequence."
    # },

    # {
    #     "id": "Q002",
    #     "question": "What types of room accommodations are available and how many beds do they have?",
    #     "ground_truth": "The hospital offers three types of rooms: a General Ward with 6-bed shared accommodation, a Semi-Private Room with 2 beds, and a Private Room with single occupancy.",
    #     "source_doc": "admission_and_discharge_process.pdf",
    #     "notes": "Checks data extraction for room types and their exact bed configurations."
    # },
    {
        "id": "Q003",
        "question": "What steps are involved in the hospital discharge process?",
        "ground_truth": "The discharge process involves four steps: physician approval, final billing, medication counseling, and follow-up appointment scheduling.",
        "source_doc": "admission_and_discharge_process.pdf",
        "notes": "Validates extraction of sequential discharge workflows."
    },
    # {
    #     "id": "Q004",
    #     "question": "How long does the estimated discharge process take after obtaining physician approval?",
    #     "ground_truth": "The estimated discharge time is 2 to 4 hours after physician approval.",
    #     "source_doc": "admission_and_discharge_process.pdf",
    #     "notes": "Tests quantitative/time-range metric retrieval."
    # },

#     # ─────────────────────────────────────────────
#     # appointment_booking_guide.pdf
#     # ─────────────────────────────────────────────
    # {
    #     "id": "Q005",
    #     "question": "What methods can patients use to book an appointment at the hospital?",
    #     "ground_truth": "Patients may book appointments through the Hospital Reception, Telephone, Hospital Website, or Mobile Application.",
    #     "source_doc": "appointment_booking_guide.pdf",
    #     "notes": "Tests listing capabilities for channels or platforms."
    # },
    # {
    #     "id": "Q006",
    #     "question": "What are the operational timings for General OP and Specialist OP?",
    #     "ground_truth": "General OP operates from 8:00 AM to 8:00 PM, and Specialist OP operates from 9:00 AM to 5:00 PM.",
    #     "source_doc": "appointment_booking_guide.pdf",
    #     "notes": "Checks specific multi-category time-frame matching."
    # },
    ]
#     {
#         "id": "Q007",
#         "question": "What is the cancellation policy timeframe for scheduled consultations?",
#         "ground_truth": "Appointments may be cancelled up to 4 hours before the scheduled consultation.",
#         "source_doc": "appointment_booking_guide.pdf",
#         "notes": "Validates precise policy/rule retrieval constraints."
#     },
#     {
#         "id": "Q008",
#         "question": "Which specific specialties or consultations support teleconsultation?",
#         "ground_truth": "Teleconsultation is available for General Medicine, Dermatology, Psychiatry, and Follow-up consultations.",
#         "source_doc": "appointment_booking_guide.pdf",
#         "notes": "Tests multi-item retrieval from bulleted lists."
#     },
#     {
#         "id": "Q009",
#         "question": "How early are patients advised to arrive before their scheduled appointment time?",
#         "ground_truth": "Patients are advised to arrive 15 minutes before their appointment time.",
#         "source_doc": "appointment_booking_guide.pdf",
#         "notes": "Tests simple numerical threshold/FAQ matching."
#     },
#     {
#         "id": "Q010",
#         "question": "Can patients reschedule their appointments?",
#         "ground_truth": "Yes, patients can reschedule their appointments.",
#         "source_doc": "appointment_booking_guide.pdf",
#         "notes": "Tests simple FAQ yes/no retrieval."
#     },

#     # ─────────────────────────────────────────────
#     # department_directory.pdf
#     # ─────────────────────────────────────────────
#     {
#         "id": "Q011",
#         "question": "Who is the head of General Medicine and what specific services do they handle?",
#         "ground_truth": "Dr. Arjun Nair is the head of General Medicine. Services include fever management, diabetes care, and hypertension management.",
#         "source_doc": "department_directory.pdf",
#         "notes": "Tests cross-linking a department head name with its services list."
#     },
#     {
#         "id": "Q012",
#         "question": "Who are the respective heads of Cardiology, Neurology, Pediatrics, and Orthopedics?",
#         "ground_truth": "The head of Cardiology is Dr. Priya Menon, Neurology is Dr. Rahul Sharma, Pediatrics is Dr. Meera Thomas, and Orthopedics is Dr. Karthik Raman.",
#         "source_doc": "department_directory.pdf",
#         "notes": "Tests multiple entity-to-name mapping lookups simultaneously."
#     },
#     {
#         "id": "Q013",
#         "question": "Which medical departments are headed by Dr. Anitha Joseph, Dr. Suresh Kumar, and Dr. Lakshmi Nair?",
#         "ground_truth": "Dr. Anitha Joseph heads Dermatology, Dr. Suresh Kumar heads ENT, and Dr. Lakshmi Nair heads Obstetrics and Gynecology.",
#         "source_doc": "department_directory.pdf",
#         "notes": "Reverse lookup: checks department identification based on doctor names."
#     },
#     {
#         "id": "Q014",
#         "question": "What services are available 24x7 according to the department directory?",
#         "ground_truth": "Emergency Medicine is available as a 24x7 service.",
#         "source_doc": "department_directory.pdf",
#         "notes": "Tests keyword search matching continuous timeline services."
#     },
#     {
#         "id": "Q015",
#         "question": "What services does the Cardiology department provide?",
#         "ground_truth": "The Cardiology department provides ECG, Echocardiogram, and Cardiac consultation services.",
#         "source_doc": "department_directory.pdf",
#         "notes": "Tests service-level detail retrieval for a specific department."
#     },
#     {
#         "id": "Q016",
#         "question": "What services does the Neurology department provide?",
#         "ground_truth": "The Neurology department provides Stroke evaluation and Epilepsy treatment.",
#         "source_doc": "department_directory.pdf",
#         "notes": "Tests service-level detail retrieval for Neurology."
#     },

#     # ─────────────────────────────────────────────
#     # diagnostic_imaging_guide.pdf
#     # ─────────────────────────────────────────────
#     {
#         "id": "Q017",
#         "question": "What are the diagnostic use cases listed for X-Ray and Ultrasound scans?",
#         "ground_truth": "X-Ray is used for bone injuries and chest evaluation. Ultrasound is used for pregnancy monitoring and abdominal evaluation.",
#         "source_doc": "diagnostic_imaging_guide.pdf",
#         "notes": "Tests category matching for basic non-contrast modalities."
#     },
#     {
#         "id": "Q018",
#         "question": "What are the listed diagnostic uses for a CT Scan?",
#         "ground_truth": "CT Scan is used for head injuries and internal organ assessment.",
#         "source_doc": "diagnostic_imaging_guide.pdf",
#         "notes": "Verifies alignment with slightly broken/misaligned PDF layouts."
#     },
#     {
#         "id": "Q019",
#         "question": "What indications require an MRI scan according to the imaging guide?",
#         "ground_truth": "An MRI scan is used for brain imaging, spine evaluation, and joint assessment.",
#         "source_doc": "diagnostic_imaging_guide.pdf",
#         "notes": "Tests specific high-end imaging lookup parameters."
#     },
#     {
#         "id": "Q020",
#         "question": "What patient preparations are required for an MRI and a CT Scan?",
#         "ground_truth": "For an MRI, patients must remove metal objects before the scan. For a CT Scan, contrast studies may require fasting.",
#         "source_doc": "diagnostic_imaging_guide.pdf",
#         "notes": "Tests distinct preparation instructions per scanner type."
#     },
#     {
#         "id": "Q021",
#         "question": "What is the expected report turnaround time for X-Rays, Ultrasounds, CT Scans, and MRIs?",
#         "ground_truth": "X-Ray and Ultrasound reports are available the same day. CT Scan reports are ready in 24 hours, and MRI reports take 24 to 48 hours.",
#         "source_doc": "diagnostic_imaging_guide.pdf",
#         "notes": "Validates complex multi-row entity metric extraction and comparison."
#     },

#     # ─────────────────────────────────────────────
#     # hospital_services_brochure.pdf
#     # ─────────────────────────────────────────────
#     {
#         "id": "Q022",
#         "question": "What core services does the hospital provide?",
#         "ground_truth": "The hospital provides Emergency Medicine with 24x7 services, Outpatient Services with daily specialist consultations, Inpatient Services including private rooms, shared rooms, and ICU care, and a 24-hour Pharmacy.",
#         "source_doc": "hospital_services_brochure.pdf",
#         "notes": "Tests broad service overview retrieval from the brochure."
#     },
#     {
#         "id": "Q023",
#         "question": "What diagnostic, surgical, and rehabilitation services are offered by the hospital?",
#         "ground_truth": "Diagnostic services include Laboratory, Radiology, Ultrasound, CT Scan, and MRI. Surgical services include General Surgery, Orthopedics, ENT Surgery, and Laparoscopic Surgery. Rehabilitation services include Physiotherapy and Occupational Therapy.",
#         "source_doc": "hospital_services_brochure.pdf",
#         "notes": "Tests multi-category service listing retrieval across three service types."
#     },
# ]
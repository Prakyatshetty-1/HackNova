export const subjectsData = {
  "Sem 1": [
    { value: "engineering-mathematics-1", label: "Engineering Mathematics-I" },
    { value: "engineering-physics-1", label: "Engineering Physics-I" },
    { value: "engineering-chemistry-1", label: "Engineering Chemistry-I" },
    { value: "engineering-mechanics", label: "Engineering Mechanics" },
    { value: "basic-electrical-engineering", label: "Basic Electrical Engineering" }
  ],
  "Sem 2": [
    { value: "engineering-mathematics-2", label: "Engineering Mathematics-II" },
    { value: "engineering-physics-2", label: "Engineering Physics-II" },
    { value: "engineering-chemistry-2", label: "Engineering Chemistry-II" },
    { value: "engineering-graphics", label: "Engineering Graphics" },
    { value: "c-programming", label: "C Programming" },
    { value: "professional-communication", label: "Professional Communication and Ethics-I" }
  ],
  "Sem 3": [
    { value: "engineering-mathematics-3", label: "Engineering Mathematics-III" },
    { value: "discrete-structures", label: "Discrete Structures and Graph Theory" },
    { value: "data-structure", label: "Data Structure" },
    { value: "digital-logic", label: "Digital Logic & Computer Architecture" },
    { value: "computer-graphics", label: "Computer Graphics" }
  ],
  "Sem 4": [
    { value: "engineering-mathematics-4", label: "Engineering Mathematics-IV" },
    { value: "analysis-of-algorithm", label: "Analysis of Algorithm" },
    { value: "database-management", label: "Database Management System" },
    { value: "operating-system", label: "Operating System" },
    { value: "microprocessor", label: "Microprocessor" }
  ],
  "Sem 5": [
    { value: "theoretical-computer-science", label: "Theoretical Computer Science" },
    { value: "software-engineering", label: "Software Engineering" },
    { value: "computer-network", label: "Computer Network" },
    { value: "data-warehousing", label: "Data Warehousing & Mining" },
    { value: "dloc-1", label: "Department Level Optional Course-1" }
  ],
  "Sem 6": [
    { value: "system-programming", label: "System Programming & Compiler Construction" },
    { value: "cryptography", label: "Cryptography & System Security" },
    { value: "mobile-computing", label: "Mobile Computing" },
    { value: "artificial-intelligence", label: "Artificial Intelligence" },
    { value: "dloc-2", label: "Department Level Optional Course-2" }
  ],
  "Sem 7": [
    { value: "machine-learning", label: "Machine Learning" },
    { value: "big-data-analysis", label: "Big Data Analysis" },
    { value: "dloc-3", label: "Department Level Optional Course-3" },
    { value: "dloc-4", label: "Department Level Optional Course-4" },
    { value: "iloc-1", label: "Institute Level Optional Course-1" }
  ],
  "Sem 8": [
    { value: "distributed-computing", label: "Distributed Computing" },
    { value: "dloc-5", label: "Department Level Optional Course-5" },
    { value: "dloc-6", label: "Department Level Optional Course-6" },
    { value: "iloc-2", label: "Institute Level Optional Course-2" }
  ]
};

// Get all subjects across all semesters
export const getAllSubjects = () => {
  return Object.values(subjectsData).flat();
};

// Get subjects for a specific semester
export const getSubjectsBySemester = (semester) => {
  return subjectsData[semester] || [];
};
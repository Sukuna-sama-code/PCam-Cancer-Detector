export const sample = {
  age: "56",
  sex: "Female",
  smoking: "Former",
  packs: "24",
  quit: "6",
  family: "Unsure",
  symptom: "None reported",
  report: "",
};
export const providers = [
  {
    name: "Manipal Hospital, Old Airport Road",
    specialty: "Breast imaging",
    service: "Breast MRI",
    city: "Bengaluru",
    initials: "MH",
    tone: "sage",
    certificate:
      "https://portal.nabh.co/Documents/AccreditedList/Hospitals/H-2008-0012_5th%20Edition.pdf",
    serviceUrl:
      "https://www.manipalhospitals.com/oldairportroad/specialities/radiology/mri/",
  },
  {
    name: "Willow Imaging Centre",
    specialty: "Thoracic imaging",
    service: "Low-dose CT",
    city: "Bengaluru",
    initials: "WI",
    tone: "sage",
  },
  {
    name: "Cedar Skin Clinic",
    specialty: "Dermatology",
    service: "Dermoscopy",
    city: "Bengaluru",
    initials: "CS",
    tone: "gold",
  },
  {
    name: "Magnolia Breast Centre",
    specialty: "Breast imaging",
    service: "Breast MRI",
    city: "Mysuru",
    initials: "MB",
    tone: "lilac",
  },
  {
    name: "Alder Family Practice",
    specialty: "Primary care",
    service: "Clinical review",
    city: "Mysuru",
    initials: "AF",
    tone: "sage",
  },
];
export function pathways(p) {
  const symptoms = p.symptom !== "None reported";
  const ct = [
    p.age !== "" && +p.age >= 50 && +p.age <= 80,
    p.packs !== "" && +p.packs >= 20,
    p.smoking === "Current" ||
      (p.smoking === "Former" && p.quit !== "" && +p.quit < 15),
    !symptoms,
  ];
  const skin = p.symptom === "Changing skin lesion",
    family = p.family === "Breast or ovarian cancer";
  return [
    {
      name: "Clinical review",
      type: "PRIMARY CARE",
      show: symptoms,
      score: symptoms ? 6 : 0,
      meter: null,
      tag: "Discuss symptoms",
      desc: "New or persistent symptoms deserve an individual assessment.",
      why: "Screening guidance applies to people without symptoms. A clinician can examine the area and decide which tests are appropriate.",
      prep: "Record when the change began and whether it is getting worse. Seek urgent help for severe or rapidly worsening symptoms.",
      source: "NCI · Screening tests",
      url: "https://www.cancer.gov/about-cancer/screening/screening-tests",
    },
    {
      name: "Low-dose CT",
      type: "LUNG SCREENING",
      score: ct.every(Boolean) ? 4 : 0,
      meter: ct.filter(Boolean).length,
      total: 4,
      tag: ct.every(Boolean) ? "Discuss screening" : "Criteria incomplete",
      desc: "A low-dose chest scan for people with a qualifying smoking history.",
      why: "US guidance covers adults aged 50–80 with at least 20 pack-years who currently smoke or quit within 15 years. This screening pathway assumes no symptoms. A clinician weighs potential benefits and harms.",
      prep: "Bring smoking dates, average packs per day and any earlier chest scans. One pack-year means one pack a day for one year.",
      source: "USPSTF · Lung cancer screening",
      url: "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/lung-cancer-screening",
    },
    {
      name: "Dermoscopy",
      type: "SKIN ASSESSMENT",
      score: skin ? 5 : 0,
      meter: null,
      tag: skin ? "Discuss skin change" : "Clinician-directed",
      desc: "A magnified skin examination for a lesion that needs assessment.",
      why: "NICE advises trained professionals to use dermoscopy when assessing referred pigmented lesions. The intake alone cannot determine whether a lesion needs a biopsy.",
      prep: "Bring dated photographs, if available, and describe changes in size, colour or bleeding.",
      source: "NICE · Melanoma assessment",
      url: "https://www.nice.org.uk/guidance/ng14/chapter/Recommendations",
    },
    {
      name: "Breast MRI",
      type: "SPECIALIST IMAGING",
      score: family ? 2 : 0,
      meter: null,
      tag: family ? "Review family history" : "Risk assessment needed",
      desc: "Additional imaging for selected people at high breast cancer risk.",
      why: "NCI describes MRI alongside mammography for some people at high risk. A family history alone does not establish eligibility.",
      prep: "Bring prior imaging and relatives’ ages at diagnosis. Ask whether formal risk assessment is appropriate.",
      source: "NCI · Breast cancer screening",
      url: "https://www.cancer.gov/types/breast/screening",
    },
    {
      name: "Liquid biopsy",
      type: "RESEARCH CONTEXT",
      score: -1,
      meter: null,
      tag: "Educational only",
      desc: "Blood-based multi-cancer detection is an active area of research.",
      why: "NCI reports that the benefit of multi-cancer tests for screening people without symptoms remains uncertain. They do not replace established screening.",
      prep: "Ask which research questions remain open and how an abnormal finding would be followed up.",
      source: "NCI · Multi-cancer detection",
      url: "https://prevention.cancer.gov/research-areas/networks-consortia-programs/csrn/q-a-about-mcd-tests",
    },
  ]
    .filter((x) => x.show !== false)
    .sort((a, b) => b.score - a.score);
}

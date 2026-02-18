/**
 * Form Templates - Define structured fields with sections/categories
 * This improves extraction accuracy by providing context to the AI
 */

export const FORM_TEMPLATES = {
    // Medical Form Template
    'medical_form': {
        name: 'Medical Form',
        sections: [
            {
                name: 'Patient',
                fields: [
                    { key: 'Patient_Name', label: 'Name', question: 'What is the name in Patient section?' },
                    { key: 'Patient_Age', label: 'Age', question: 'What is the age in Patient section?' },
                    { key: 'Patient_DOB', label: 'Date of Birth', question: 'What is the date of birth in Patient section?' },
                    { key: 'Patient_Gender', label: 'Gender', question: 'What is the gender in Patient section?' }
                ]
            },
            {
                name: 'Insurance',
                fields: [
                    { key: 'Insurance_Company', label: 'Company', question: 'What is the insurance company name?' },
                    { key: 'Insurance_Policy', label: 'Policy Number', question: 'What is the insurance policy number?' },
                    { key: 'Insurance_Claim', label: 'Claim Number', question: 'What is the claim number?' }
                ]
            }
        ]
    },

    // CPL Form 2 Template
    'cpl_form_2': {
        name: 'CPL Form 2',
        sections: [
            {
                name: 'Client',
                fields: [
                    { key: 'Client_Name', label: 'Name', question: 'What is the name in Client section?' },
                    { key: 'Client_DOB', label: 'Date of Birth', question: 'What is the date of birth in Client section?' },
                    { key: 'Client_Phone', label: 'Phone', question: 'What is the phone number in Client section?' },
                    { key: 'Client_Address', label: 'Address', question: 'What is the address in Client section?' }
                ]
            },
            {
                name: 'Order',
                fields: [
                    { key: 'Order_Billing_Type', label: 'Billing Type', question: 'What is the billing type in Order section?' },
                    { key: 'Order_Date', label: 'Order Date', question: 'What is the order date in Order section?' },
                    { key: 'Order_Number', label: 'Order Number', question: 'What is the order number in Order section?' }
                ]
            },
            {
                name: 'Patient',
                fields: [
                    { key: 'Patient_Name', label: 'Name', question: 'What is the name in Patient section?' },
                    { key: 'Patient_Age', label: 'Age', question: 'What is the age in Patient section?' },
                    { key: 'Patient_Gender', label: 'Gender', question: 'What is the gender in Patient section?' }
                ]
            },
            {
                name: 'Primary Insurance',
                fields: [
                    { key: 'Primary_Insurance_Name', label: 'Insurance Name', question: 'What is the insurance company name in Primary Insurance section?' },
                    { key: 'Primary_Insurance_Policy', label: 'Policy Number', question: 'What is the policy number in Primary Insurance section?' },
                    { key: 'Primary_Insurance_Group', label: 'Group Number', question: 'What is the group number in Primary Insurance section?' },
                    { key: 'Primary_Insurance_Phone', label: 'Phone', question: 'What is the phone number in Primary Insurance section?' }
                ]
            },
            {
                name: 'Secondary Insurance',
                fields: [
                    { key: 'Secondary_Insurance_Name', label: 'Insurance Name', question: 'What is the insurance company name in Secondary Insurance section?' },
                    { key: 'Secondary_Insurance_Policy', label: 'Policy Number', question: 'What is the policy number in Secondary Insurance section?' },
                    { key: 'Secondary_Insurance_Group', label: 'Group Number', question: 'What is the group number in Secondary Insurance section?' }
                ]
            },
            {
                name: 'Guarantor',
                fields: [
                    { key: 'Guarantor_Name', label: 'Name', question: 'What is the name in Guarantor section?' },
                    { key: 'Guarantor_Relationship', label: 'Relationship', question: 'What is the relationship to patient in Guarantor section?' },
                    { key: 'Guarantor_Phone', label: 'Phone', question: 'What is the phone number in Guarantor section?' }
                ]
            },
            {
                name: 'Physician',
                fields: [
                    { key: 'Physician_Name', label: 'Name', question: 'What is the physician name in Physician section?' },
                    { key: 'Physician_NPI', label: 'NPI', question: 'What is the NPI number in Physician section?' },
                    { key: 'Physician_Phone', label: 'Phone', question: 'What is the phone number in Physician section?' }
                ]
            }
        ]
    },

    // Generic/Custom Template
    'custom': {
        name: 'Custom Fields',
        sections: [
            {
                name: 'General',
                fields: []
            }
        ]
    }
};

/**
 * Get all fields from a template as a flat array
 */
export function getTemplateFields(templateId) {
    const template = FORM_TEMPLATES[templateId];
    if (!template) return [];

    const allFields = [];
    template.sections.forEach(section => {
        section.fields.forEach(field => {
            allFields.push({
                key: field.key,
                question: field.question,
                section: section.name,
                label: field.label
            });
        });
    });

    return allFields;
}

/**
 * Get template sections for UI display
 */
export function getTemplateSections(templateId) {
    const template = FORM_TEMPLATES[templateId];
    return template ? template.sections : [];
}

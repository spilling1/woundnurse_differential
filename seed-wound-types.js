import { storage } from "./server/storage.js";

const defaultWoundTypes = [
  {
    name: 'general',
    displayName: 'General Instructions',
    description: 'Default instructions for wound assessment and care',
    instructions: 'Provide comprehensive wound assessment including wound type identification, staging, measurement, and appropriate care recommendations.',
    isEnabled: true,
    isDefault: true,
    priority: 100
  },
  {
    name: 'pressure_injury',
    displayName: 'Pressure Injury (Ulcer)',
    description: 'Pressure ulcers and bedsores',
    instructions: 'Assess for pressure injury staging (Stage 1-4, unstageable, suspected deep tissue injury). Consider pressure relief, positioning, support surfaces, and nutrition.',
    isEnabled: true,
    isDefault: false,
    priority: 90
  },
  {
    name: 'venous_ulcer',
    displayName: 'Venous Ulcer',
    description: 'Chronic venous insufficiency wounds',
    instructions: 'Assess for venous insufficiency signs, edema, and hemosiderin staining. Recommend compression therapy and elevation.',
    isEnabled: true,
    isDefault: false,
    priority: 80
  },
  {
    name: 'arterial_ulcer',
    displayName: 'Arterial Insufficiency Ulcer',
    description: 'Arterial insufficiency and ischemic wounds',
    instructions: 'Assess for arterial insufficiency, pale wound bed, and poor perfusion. Urgent vascular evaluation may be needed.',
    isEnabled: true,
    isDefault: false,
    priority: 70
  },
  {
    name: 'diabetic_ulcer',
    displayName: 'Diabetic Ulcer',
    description: 'Diabetic foot ulcers and neuropathic wounds',
    instructions: 'Assess for neuropathy, infection, and vascular compromise. Recommend glucose control, offloading, and diabetic foot care.',
    isEnabled: true,
    isDefault: false,
    priority: 60
  },
  {
    name: 'surgical_wound',
    displayName: 'Surgical Wound',
    description: 'Post-operative and surgical site wounds',
    instructions: 'Assess healing progress, signs of infection, and dehiscence. Follow post-operative care protocols.',
    isEnabled: true,
    isDefault: false,
    priority: 50
  },
  {
    name: 'traumatic_wound',
    displayName: 'Traumatic Wound',
    description: 'Acute traumatic injuries and lacerations',
    instructions: 'Assess for foreign bodies, contamination, and tissue damage. Consider tetanus prophylaxis and wound irrigation.',
    isEnabled: true,
    isDefault: false,
    priority: 40
  },
  {
    name: 'ischemic_wound',
    displayName: 'Ischemic Wound',
    description: 'Wounds caused by insufficient blood flow',
    instructions: 'Assess for ischemia, tissue necrosis, and vascular compromise. Urgent vascular evaluation required.',
    isEnabled: true,
    isDefault: false,
    priority: 30
  },
  {
    name: 'radiation_wound',
    displayName: 'Radiation Wound',
    description: 'Radiation therapy-related skin damage',
    instructions: 'Assess for radiation dermatitis and delayed healing. Gentle care and moisture management recommended.',
    isEnabled: true,
    isDefault: false,
    priority: 20
  },
  {
    name: 'infectious_wound',
    displayName: 'Infectious Wound',
    description: 'Wounds with active infection',
    instructions: 'Assess for signs of infection, purulence, and systemic symptoms. Consider antibiotic therapy and wound culture.',
    isEnabled: true,
    isDefault: false,
    priority: 10
  }
];

async function seedWoundTypes() {
  try {
    for (const woundType of defaultWoundTypes) {
      const existing = await storage.getWoundTypeByName(woundType.name);
      if (!existing) {
        await storage.createWoundType(woundType);
        console.log('Created wound type:', woundType.displayName);
      }
    }
    console.log('Wound types seeded successfully');
  } catch (error) {
    console.error('Error seeding wound types:', error);
  }
  process.exit(0);
}

seedWoundTypes();
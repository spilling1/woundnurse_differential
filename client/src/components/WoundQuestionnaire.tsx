import { useState } from "react";
import { HelpCircle, ChevronDown, ChevronUp } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface WoundQuestionnaireProps {
  onDataChange: (data: WoundContextData) => void;
  initialData?: WoundContextData;
}

export interface WoundContextData {
  woundOrigin: string;
  medicalHistory: string;
  woundChanges: string;
  currentCare: string;
  woundPain: string;
  supportAtHome: string;
  mobilityStatus: string;
  nutritionStatus: string;
  stressLevel: string;
  comorbidities: string;
  age: string;
  obesity: string;
  medications: string;
  alcoholUse: string;
  smokingStatus: string;
  frictionShearing: string;
  knowledgeDeficits: string;
  woundSite: string;
}

export default function WoundQuestionnaire({ onDataChange, initialData }: WoundQuestionnaireProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [data, setData] = useState<WoundContextData>(initialData || {
    woundOrigin: '',
    medicalHistory: '',
    woundChanges: '',
    currentCare: '',
    woundPain: '',
    supportAtHome: '',
    mobilityStatus: '',
    nutritionStatus: '',
    stressLevel: '',
    comorbidities: '',
    age: '',
    obesity: '',
    medications: '',
    alcoholUse: '',
    smokingStatus: '',
    frictionShearing: '',
    knowledgeDeficits: '',
    woundSite: ''
  });

  const handleChange = (field: keyof WoundContextData, value: string) => {
    const newData = { ...data, [field]: value };
    setData(newData);
    onDataChange(newData);
  };

  const questions = [
    {
      key: 'woundSite' as keyof WoundContextData,
      label: 'Site of Wound',
      placeholder: 'e.g., "Right heel", "Lower back", "Abdomen"',
      required: true,
      rows: 1
    },
    {
      key: 'woundOrigin' as keyof WoundContextData,
      label: 'How and when did this wound occur?',
      placeholder: 'e.g., "Pressure ulcer from prolonged sitting, noticed 3 days ago"',
      required: true,
      rows: 2
    },
    {
      key: 'age' as keyof WoundContextData,
      label: 'Age',
      placeholder: 'e.g., "65", "Adult", "Elderly"',
      required: true,
      rows: 1
    },
    {
      key: 'comorbidities' as keyof WoundContextData,
      label: 'Comorbidities and Medical Conditions',
      placeholder: 'e.g., "Type 2 diabetes, hypertension, heart disease" or "None"',
      required: true,
      rows: 2
    },
    {
      key: 'medications' as keyof WoundContextData,
      label: 'Current Medications',
      placeholder: 'e.g., "Metformin, blood thinners, steroids" or "None"',
      required: true,
      rows: 2
    },
    {
      key: 'nutritionStatus' as keyof WoundContextData,
      label: 'Nutritional Status',
      placeholder: 'e.g., "Poor appetite, weight loss" or "Good nutrition"',
      rows: 1
    },
    {
      key: 'obesity' as keyof WoundContextData,
      label: 'Weight Status/Obesity',
      placeholder: 'e.g., "Obese (BMI 35)", "Normal weight", "Unknown"',
      rows: 1
    },
    {
      key: 'mobilityStatus' as keyof WoundContextData,
      label: 'Mobility Status',
      placeholder: 'e.g., "Wheelchair user", "Bedridden", "Active"',
      rows: 1
    },
    {
      key: 'frictionShearing' as keyof WoundContextData,
      label: 'Friction/Shearing Risk',
      placeholder: 'e.g., "High risk - sliding in bed", "Low risk"',
      rows: 1
    },
    {
      key: 'smokingStatus' as keyof WoundContextData,
      label: 'Smoking Status',
      placeholder: 'e.g., "Current smoker - 1 pack/day", "Never smoked"',
      rows: 1
    },
    {
      key: 'alcoholUse' as keyof WoundContextData,
      label: 'Alcohol Use',
      placeholder: 'e.g., "Heavy drinker", "Occasional", "No alcohol"',
      rows: 1
    },
    {
      key: 'stressLevel' as keyof WoundContextData,
      label: 'Stress Level',
      placeholder: 'e.g., "High stress", "Moderate", "Low stress"',
      rows: 1
    },
    {
      key: 'woundChanges' as keyof WoundContextData,
      label: 'Recent changes in the wound',
      placeholder: 'e.g., "Increased redness and drainage" or "Healing well"',
      rows: 2
    },
    {
      key: 'currentCare' as keyof WoundContextData,
      label: 'Current wound care routine',
      placeholder: 'e.g., "Daily bandage change with antibiotic ointment"',
      rows: 2
    },
    {
      key: 'woundPain' as keyof WoundContextData,
      label: 'Pain level and description',
      placeholder: 'e.g., "Constant throbbing pain, 7/10"',
      rows: 1
    },
    {
      key: 'knowledgeDeficits' as keyof WoundContextData,
      label: 'Knowledge Deficits',
      placeholder: 'e.g., "Unsure about proper care", "Good understanding"',
      rows: 1
    },
    {
      key: 'supportAtHome' as keyof WoundContextData,
      label: 'Support available at home',
      placeholder: 'e.g., "Spouse helps with dressing changes"',
      rows: 1
    }
  ];

  const requiredQuestions = questions.filter(q => q.required);
  const optionalQuestions = questions.filter(q => !q.required);

  return (
    <Card className="mb-6">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <HelpCircle className="text-medical-blue text-lg mr-2" />
            <h2 className="text-lg font-semibold text-gray-900">Wound Context Questions</h2>
          </div>
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? (
              <>
                <ChevronUp className="mr-2 h-4 w-4" />
                Collapse
              </>
            ) : (
              <>
                <ChevronDown className="mr-2 h-4 w-4" />
                Expand All
              </>
            )}
          </Button>
        </div>

        <p className="text-sm text-gray-600 mb-6">
          Providing additional context helps the AI generate more personalized and accurate care plans. 
          Required questions help ensure safety, while optional questions provide deeper insights.
        </p>

        {/* Required Questions */}
        <div className="space-y-4 mb-6">
          <h3 className="font-medium text-gray-900 text-sm">Essential Information</h3>
          {requiredQuestions.map((question) => (
            <div key={question.key} className="space-y-2">
              <Label className="text-sm font-medium text-gray-700 flex items-center">
                {question.label}
                <span className="text-red-500 ml-1">*</span>
              </Label>
              <Textarea
                value={data[question.key]}
                onChange={(e) => handleChange(question.key, e.target.value)}
                placeholder={question.placeholder}
                rows={question.rows || 2}
                className="text-sm"
              />
            </div>
          ))}
        </div>

        {/* Optional Questions - Collapsible */}
        <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" className="w-full justify-between p-0 h-auto">
              <h3 className="font-medium text-gray-900 text-sm">Additional Context (Optional but will improve your results)</h3>
              {isExpanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          </CollapsibleTrigger>
          
          <CollapsibleContent className="space-y-4 mt-4">
            {optionalQuestions.map((question) => (
              <div key={question.key} className="space-y-2">
                <Label className="text-sm font-medium text-gray-700">
                  {question.label}
                </Label>
                <Textarea
                  value={data[question.key]}
                  onChange={(e) => handleChange(question.key, e.target.value)}
                  placeholder={question.placeholder}
                  rows={question.rows || 2}
                  className="text-sm"
                />
              </div>
            ))}
          </CollapsibleContent>
        </Collapsible>

        <div className="mt-4 p-3 bg-blue-50 rounded-lg">
          <p className="text-xs text-blue-800">
            <strong>Privacy:</strong> This information is used only to generate your care plan and is not stored permanently. 
            It helps the AI understand your specific situation for more relevant recommendations.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
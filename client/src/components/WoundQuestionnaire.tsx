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
    nutritionStatus: ''
  });

  const handleChange = (field: keyof WoundContextData, value: string) => {
    const newData = { ...data, [field]: value };
    setData(newData);
    onDataChange(newData);
  };

  const questions = [
    {
      key: 'woundOrigin' as keyof WoundContextData,
      label: 'How and when did this wound occur?',
      placeholder: 'e.g., "Pressure ulcer from prolonged sitting, noticed 3 days ago" or "Cut from kitchen knife yesterday"',
      required: true
    },
    {
      key: 'medicalHistory' as keyof WoundContextData,
      label: 'Relevant medical history and current medications',
      placeholder: 'e.g., "Type 2 diabetes, taking blood thinners, history of slow healing" or "Generally healthy, no medications"',
      required: true
    },
    {
      key: 'woundChanges' as keyof WoundContextData,
      label: 'Any recent changes in the wound?',
      placeholder: 'e.g., "Increased redness and warmth, foul odor, more drainage today" or "Healing well, less pain"'
    },
    {
      key: 'currentCare' as keyof WoundContextData,
      label: 'Current wound care routine',
      placeholder: 'e.g., "Changing bandage daily with antibiotic ointment" or "Just keeping it clean and dry"'
    },
    {
      key: 'woundPain' as keyof WoundContextData,
      label: 'Pain level and description',
      placeholder: 'e.g., "Constant throbbing pain, 7/10" or "Mild discomfort when touched, 3/10"'
    },
    {
      key: 'supportAtHome' as keyof WoundContextData,
      label: 'Support available at home',
      placeholder: 'e.g., "Spouse helps with dressing changes" or "Living alone, managing independently"'
    },
    {
      key: 'mobilityStatus' as keyof WoundContextData,
      label: 'Mobility and activity level',
      placeholder: 'e.g., "Wheelchair user, limited mobility" or "Active, walking daily"'
    },
    {
      key: 'nutritionStatus' as keyof WoundContextData,
      label: 'Nutrition and appetite',
      placeholder: 'e.g., "Poor appetite, recent weight loss" or "Eating well, maintaining weight"'
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
                rows={2}
                className="text-sm"
              />
            </div>
          ))}
        </div>

        {/* Optional Questions - Collapsible */}
        <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" className="w-full justify-between p-0 h-auto">
              <h3 className="font-medium text-gray-900 text-sm">Additional Context (Optional)</h3>
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
                  rows={2}
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
export function generateCaseId(): string {
  const date = new Date();
  const dateStr = date.toISOString().split('T')[0];
  const counter = String(Math.floor(Math.random() * 1000)).padStart(3, '0');
  return `${dateStr}-${counter}`;
}

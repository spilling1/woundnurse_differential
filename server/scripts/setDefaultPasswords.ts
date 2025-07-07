import bcrypt from 'bcrypt';
import { db } from '../db';
import { users } from '@shared/schema';
import { eq, isNull } from 'drizzle-orm';

const DEFAULT_PASSWORD = 'Woundnurse';

async function setDefaultPasswords() {
  console.log('Setting default passwords for existing users...');
  
  try {
    // Get all users without passwords
    const usersWithoutPassword = await db.select().from(users).where(isNull(users.password));
    
    console.log(`Found ${usersWithoutPassword.length} users without passwords`);
    
    // Hash the default password
    const hashedPassword = await bcrypt.hash(DEFAULT_PASSWORD, 10);
    
    // Update each user
    for (const user of usersWithoutPassword) {
      await db.update(users)
        .set({ 
          password: hashedPassword,
          mustChangePassword: true 
        })
        .where(eq(users.id, user.id));
      
      console.log(`Updated password for user: ${user.email}`);
    }
    
    console.log('Default passwords set successfully!');
    console.log(`Default password: ${DEFAULT_PASSWORD}`);
    console.log('All users must change their password on first login.');
    
  } catch (error) {
    console.error('Error setting default passwords:', error);
  }
}

// Run the script
setDefaultPasswords();
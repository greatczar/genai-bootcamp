import { Amplify } from 'aws-amplify';
import {
	signIn,
	confirmSignIn,
	signOut,
	fetchAuthSession
} from '@aws-amplify/auth';

let configured = false;

export async function configureAmplify() {
	if (configured) {
		return;
	}
	// @ts-ignore
	const { env } = await import(`${''}/_app/env.js`);
	const userPoolId = env.PUBLIC_VITE_COGNITO_USER_POOL_ID;
	const userPoolClientId = env.PUBLIC_VITE_COGNITO_USER_POOL_CLIENT_ID;
	console.log('Configuring Amplify with User Pool ID:', userPoolId, 'and Client ID:', userPoolClientId);
	Amplify.configure({
		Auth: {
			Cognito: {
				userPoolId: userPoolId,
				userPoolClientId: userPoolClientId
			}
		}
	});
	configured = true;
}

export async function checkIsAuthenticated(): Promise<boolean> {
	await configureAmplify();
	try {
		// Using forceRefresh: false to use cached session if available.
		// This is faster and avoids unnecessary network requests.
		const session = await fetchAuthSession({ forceRefresh: false });
		return !!session.tokens?.idToken;
	} catch (error) {
		return false;
	}
}

export async function startPasswordlessSignIn(email: string) {
	await configureAmplify();
	try {
		const output = await signIn({
			username: email,
			options: {
				authFlowType: 'USER_AUTH'
			}
		});
    console.log('Sign in started, next step:', output.nextStep);
		return output.nextStep.signInStep === 'CONFIRM_SIGN_IN_WITH_EMAIL_CODE';
	} catch (error) {
		console.error('Error starting sign in:', error);
		throw error;
	}
}

export async function completePasswordlessSignIn(code: string) {
	await configureAmplify();
	try {
		const output = await confirmSignIn({
			challengeResponse: code
		});
		return output.isSignedIn;
	} catch (error) {
		console.error('Error confirming sign in:', error);
		throw error;
	}
}

export async function signOutUser() {
	await configureAmplify();
	try {
		await signOut();
	} catch (error) {
		console.error('Error signing out:', error);
	}
}

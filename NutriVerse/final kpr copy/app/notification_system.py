# app/notification_system.py

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def notify_ashram(ashram, donation_details):
    """
    Simulate notification to the selected ashram (can be extended to email/SMS).
    """
    print(f"\n📣 NOTIFICATION SENT TO ASHRAM: {ashram['name']}")
    print(f"📍 Location: ({ashram['latitude']}, {ashram['longitude']})")
    print("🍽️ Donation Details:")
    print(f" - Hotel: {donation_details['hotel_address']}")
    print(f" - Servings: {donation_details['amount']} for {donation_details['people']} people")
    print(f" - Type: {donation_details['food_type']}")
    print(f" - Preferred Time: {donation_details['preferred_time']}")
    print("✅ Ashram successfully notified.\n")

def notify_volunteers(volunteers_data, hotel_location, donation_details, radius_km=5):
    """
    Notify nearby volunteers (within radius_km) to assist with food pickup.
    Returns list of notification messages to display in Streamlit.
    """
    from geopy.distance import geodesic

    messages = ["🔔 **Searching for nearby volunteers...**\n"]
    for volunteer in volunteers_data:
        vol_location = (volunteer['latitude'], volunteer['longitude'])
        distance = geodesic(hotel_location, vol_location).km

        if distance <= radius_km:
            message = (
                f"✅ **Volunteer:** {volunteer['name']} (📞 {volunteer.get('phone', 'N/A')})\n"
                f" - 📍 **Distance:** {round(distance, 2)} km\n"
                f" - 🚚 **Task:** Pickup donation from {donation_details['hotel_address']}\n"
            )
            messages.append(message)
    return messages


def send_email_notification(to_email, subject, body):
    """
    Optional: Send an email (requires SMTP credentials).
    """
    from_email = "your_email@example.com"
    password = "your_password"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(from_email, password)
            server.send_message(msg)
            print(f"📧 Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Email failed: {e}")

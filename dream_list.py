import json

print("==== WELCOME TO THE DREAM LIST NOTE BOOK ====")
print("1. Add New Dream")
print("2. Watch The Dream List")
print("3. Remove Any Dream Frome You List")
print("4. exit \n")



while True:
    try:
        number_selection = int(input("\n📋 Select a number between (1-4): "))

        if number_selection == 1:
            user_input = input("\nWhat's your Dream❓: ")

            try:
                with open("Dream.json", "r") as dream_file:
                    dream_list = json.load(dream_file)
            except (FileNotFoundError, json.JSONDecodeError):
                dream_list = {}

            dream_no = f"Dream_no_{len(dream_list) + 1}"
            dream_list[dream_no] = user_input

            with open("Dream.json", "w") as dream_file:
                json.dump(dream_list, dream_file, indent=4)

            print(f"\n✅ Successfully added your Dream in list: {user_input}\n")


        elif number_selection == 2:
            try:
                with open("Dream.json","r") as dream:
                    dreams = dict(json.load(dream))
                    print(f"\nYour Dreams are:\n{", ".join(dreams.values())}")

            except (FileNotFoundError):
                print(f"\n❌ there was no file. please first make a Dream list❗\n")


        elif number_selection == 3:
            try:
                with open("Dream.json","r") as dreamss:
                    dream_list = dict(json.load(dreamss))


                    print("\n🌙 Your Dream List:")
                    print("="*50)
                    for key, value in dream_list.items():
                        print(f"✨ {key}: {value}\n")
                    print("="*50,"\n")


                    remove_key = int(input("\n📝 enter the number wich one you want to remove frome the list: "))
                    dream_key = f"Dream_no_{remove_key}"

                if dream_key in dream_list:
                    del dream_list[dream_key]
                    print(f"\n❌ Removed {remove_key} successfully!")

                else:
                    print("⚠️ Dream not found!")

                
                with open("Dream.json","w") as files:
                    json.dump(dream_list,files, indent=4)
 
            except FileNotFoundError:
                print("please first make a dream list")


        elif number_selection == 4:
            print("\n👋 Exiting program... Goodbye!\n")
            break

    except ValueError:
        print("\n❌ Please Enter Valid Numbers❗\n")
        continue
